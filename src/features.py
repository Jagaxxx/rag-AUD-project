# Handles text segmentation (chunking) and converts text chunks into vector embeddings using pre-trained sentence-transformer models.


import re
from sentence_transformers import SentenceTransformer


def chunk_text(records, chunk_size=250, overlap=40):
    """
    Group paragraphs into chunks while preserving document metadata.
    Chunks do not cross source, chapter or section boundaries.
    """

    chunks = []
    grouped = {}

    for record in records:
        key = (
            record["source"],
            record["chapter"],
            record["section"]
        )
        grouped.setdefault(key, []).append(record)

    for key, paragraphs in grouped.items():
        sentences = []

        for paragraph in paragraphs:
            parts = re.split(
                r'(?<=[.!?])\s+',
                paragraph["text"]
            )
            for sentence in parts:
                if sentence.strip():
                    sentences.append({
                        "text": sentence.strip(),
                        "page": paragraph["pdf_page"]
                    })

        start = 0

        while start < len(sentences):
            end = start
            word_count = 0

            while end < len(sentences):
                sentence_words = len(
                    sentences[end]["text"].split()
                )

                if word_count + sentence_words > chunk_size and end > start:
                    break

                word_count += sentence_words
                end += 1

                if word_count >= chunk_size:
                    break

            selected = sentences[start:end]

            if not selected:
                break

            chunks.append({
                "chunk_id": len(chunks),
                "source": key[0],
                "chapter": key[1],
                "section": key[2],
                "pdf_page_start": selected[0]["page"],
                "pdf_page_end": selected[-1]["page"],
                "text": " ".join(
                    s["text"] for s in selected
                )
            })

            if end >= len(sentences):
                break

            # Keep previous sentences for context.
            next_start = end
            overlap_words = 0

            while next_start > start:
                candidate = next_start - 1
                candidate_words = len(
                    sentences[candidate]["text"].split()
                )

                if overlap_words + candidate_words > overlap:
                    break

                overlap_words += candidate_words
                next_start -= 1

            start = next_start if next_start > start else end

    return chunks


def generate_embeddings(chunks, config):
    """Generate normalized embeddings for document chunks."""

    model = SentenceTransformer(
        config["model_name"]
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=config["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config["normalize_embeddings"]
    )

    return embeddings.astype("float32")

