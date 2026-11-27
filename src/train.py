from pathlib import Path

from src.data import load_config, get_processed_corpus
from src.features import chunk_text, generate_embeddings
from src.models import build_faiss_index, save_faiss_index


def main():
    # Load configuration
    cfg = load_config()

    # Load processed corpus
    records = get_processed_corpus(cfg)

    # Create chunks
    chunks = chunk_text(
        records,
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["chunk_overlap"]
    )

    print(f"Number of chunks: {len(chunks)}")

    # Generate embeddings
    embeddings = generate_embeddings(chunks, cfg["embeddings"])

    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(
        embeddings,
        cfg["vector_search"]
    )

    print(f"FAISS index size: {index.ntotal}")

    # Save index and chunks
    output_dir = Path(cfg["paths"]["models_dir"]) / cfg["data"]["variant"]

    save_faiss_index(index, chunks, output_dir)

    print(f"Index saved to: {output_dir}")


if __name__ == "__main__":
    main()