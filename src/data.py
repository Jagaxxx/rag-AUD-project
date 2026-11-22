# Responsible for reading raw document files from data/raw/ and performing initial text cleaning.

import json
import re
import unicodedata
from pathlib import Path
import pymupdf
import yaml

RUNNING_HEADER = re.compile(
    r"^(?:(?P<left>\d+)\s+)?(?P<num>\d+(?:\.\d+)?)\s*•\s*(?P<title>.+?)(?:\s+(?P<right>\d+))?$"
)

def load_config(path: str = "configs/config.yaml") -> dict:
    """Read the YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dictionaries to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def parse_running_header(header: str) -> dict:
    """Parse a running page header into chapter / section / printed page."""
    match = RUNNING_HEADER.match(header)
    if not match:
        return {}
    num, title = match["num"], match["title"].strip()
    page = match["left"] or match["right"]
    meta = {"chapter": num.split(".")[0], "printed_page": int(page) if page else None}
    if "." in num:
        meta["section"] = f"{num} {title}"
    return meta

def is_noise_block(text: str, noise_labels: list[str]) -> bool:
    """Decide whether a text block is layout noise rather than content."""
    flat = " ".join(text.split())
    if flat.upper() in {label.upper() for label in noise_labels}:
        return True
    if re.match(r"^(FIGURE|TABLE)\s+\d+\.\d+\b", flat):
        return True
    if re.fullmatch(r"\d{1,4}", flat):
        return True
    if flat.lower().startswith("access for free at"):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(re.match(r"\d+\.\d+\s", line) for line in lines)

def clean_paragraph(text: str) -> str:
    """Clean the text of a single block into one tidy paragraph."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    text = re.sub(r"(\w)-\n([A-Z])", r"\1-\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s*\(https?://[^)]*\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?<=[a-z][.,;”])\d{1,2}(?=\s)", "", text)
    text = re.sub(r"^INTRODUCTION\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def table_to_paragraphs(rows: list[list[str | None]]) -> list[str]:
    """Convert a table into one readable sentence-like paragraph per row with proper spacing."""
    cleaned_rows = []
    for row in rows:
        cleaned_row = []
        for cell in row:
            if cell:
                fixed_cell = " ".join(cell.split())
                cleaned_row.append(fixed_cell)
            else:
                cleaned_row.append("")
        cleaned_rows.append(cleaned_row)

    rows = cleaned_rows
    while len(rows) > 1 and not all(rows[0]):
        rows.pop(0)
    if len(rows) < 2:
        return []
    
    header, body = rows[0], rows[1:]
    paragraphs = []
    for row in body:
        pairs = [f"{h}: {c}" for h, c in zip(header, row) if h and c]
        if pairs:
            paragraphs.append("; ".join(pairs))
    return paragraphs

def extract_paragraphs(pdf_path: str | Path, top_margin: float, bottom_margin: float, noise_labels: list[str]) -> list[dict]:
    """Read the PDF and return cleaned paragraphs with metadata."""
    records = []
    source = Path(pdf_path).stem
    chapter, section = None, None

    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            r = page.rect
            header_rect = pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + top_margin)
            body_rect = pymupdf.Rect(r.x0, r.y0 + top_margin, r.x1, r.y1 - bottom_margin)

            header = parse_running_header(" ".join(page.get_text("text", clip=header_rect).split()))
            if header:
                if header["chapter"] != chapter:
                    section = None
                chapter = header["chapter"]
                section = header.get("section", section)
            else:
                chapter = section = None

            tables = [
                t for t in page.find_tables(clip=body_rect).tables
                if t.row_count >= 2 and t.col_count >= 2
            ]
            boxes = [pymupdf.Rect(t.bbox) for t in tables]

            items = []
            for block in page.get_text("blocks", clip=body_rect):
                if block[6] != 0:
                    continue
                centre = pymupdf.Point((block[0] + block[2]) / 2, (block[1] + block[3]) / 2)
                if any(box.contains(centre) for box in boxes):
                    continue
                items.append((block[1], block[4]))
            for table, box in zip(tables, boxes):
                items += [(box.y0, row) for row in table_to_paragraphs(table.extract())]
            items.sort(key=lambda item: item[0])

            for _, text in items:
                if is_noise_block(text, noise_labels):
                    continue
                cleaned = clean_paragraph(text)
                if cleaned:
                    records.append({
                        "source": source,
                        "chapter": chapter,
                        "section": section,
                        "part": "body",
                        "pdf_page": page_no,
                        "printed_page": header.get("printed_page"),
                        "text": cleaned,
                    })
    return records

def merge_page_breaks(records: list[dict]) -> list[dict]:
    """Join paragraphs that were split by a page break."""
    merged = []
    for rec in records:
        prev = merged[-1] if merged else None
        if (
            prev
            and rec["pdf_page"] == prev["pdf_page"] + 1
            and not re.search(r'[.!?:;"”)]$', prev["text"])
            and re.match(r"[a-z(]", rec["text"])
        ):
            prev["text"] += " " + rec["text"]
        else:
            merged.append(dict(rec))
    return merged

def variant_dir(cfg: dict, variant: str | None = None) -> Path:
    """Folder in data/interim/ that holds the files of one variant."""
    return Path(cfg["paths"]["interim_dir"]) / (variant or cfg["data"]["variant"])

def get_processed_corpus(cfg: dict, variant: str | None = None) -> list[dict]:
    """Process the selected PDF variant and save corpus.jsonl to data/interim/<variant>/."""
    data_cfg = cfg["data"]
    variant = variant or data_cfg["variant"]

    pdf_filename = data_cfg["variants"][variant]["pdf_file"]
    
    pdf_path = Path(cfg["paths"]["raw_dir"]) / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"There is no existing PDF: {pdf_path} ")

    records = extract_paragraphs(
        pdf_path, data_cfg["top_margin"], data_cfg["bottom_margin"], data_cfg["noise_labels"]
    )
    corpus = merge_page_breaks(records)

    out_dir = variant_dir(cfg, variant)
    save_jsonl(corpus, out_dir / "corpus.jsonl")
    return corpus