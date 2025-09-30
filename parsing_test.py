import os
from pathlib import Path
from typing import Union

from chunking.length_chunker import LengthChunker
from chunking.structure_chunker import StructureChunker
from chunking.text_chunker import TextStructureChunker
from document_parsing.grobid_parser import GROBIDParser
from document_parsing.mineru_parser import MinerUParser
from document_parsing.pymupdf_parser import PyMuPDFParser
from document_parsing.pymupdf_tesseract import PyMuPDFTesseractParser
from document_parsing.vlm_gemini import GeminiParser
from document_parsing.vlm_qwen import QwenParser
from dotenv import find_dotenv, load_dotenv
from embeddings.bge_embedder import BGEEmbedder
from embeddings.e5_embedder import E5Embedder
from embeddings.openai_embedder import OpenAIEmbedder
from retrieval.cross_encoder_retriever import CrossEncoderRetriever
from retrieval.rcs_retriever import RCSRetriever
from retrieval.token_budget_retriever import TokenBudgetRetriever
from retrieval.topk_retriever import TopKRetriever
from vectorstore.numpy_store import NumpyVectorStore

# from retrieval.token_budget_retriever import TokenBudgetRetriever
# from retrieval.rcs_retriever import RCSRetriever
# from retrieval.cross_encoder_retriever import CrossEncoderRetriever

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def run_parsers(file_path: str):
    parsers = [
        PyMuPDFParser(),
        PyMuPDFTesseractParser(),
        GROBIDParser(),
        GeminiParser(api_key=GOOGLE_API_KEY),
        QwenParser(api_key=TOGETHER_API_KEY),
        MinerUParser(),
    ]

    for parser in parsers:
        print("=" * 40)
        print(f"Parser: {parser.__class__.__name__}")
        paper_name = Path(file_path).stem
        final_base = Path("data/parsed/")

        try:
            text = parser.parse(file_path)
            out_dir = final_base / parser.__class__.__name__
            out_dir.mkdir(parents=True, exist_ok=True)  # ✅ create folder if missing

            # Save as Markdown (you could also save JSON if needed)
            out_file = out_dir / f"{paper_name}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved output to {out_file}")
            print(text[:500])  # Preview first 500 chars

        except Exception as e:
            print(f"Error: {e}")


def run_chunkers(file_path: str):
    chunkers = [
        LengthChunker(chunk_size=500, chunk_overlap=10),
        # StructureChunker(chunk_size=500, chunk_overlap=10),
        # TextStructureChunker(chunk_size=500, chunk_overlap=10),
    ]

    for chunker in chunkers:
        print("=" * 40)
        print(f"Chunker: {chunker.__class__.__name__}")
        # out_dir = Path(
        #     f"../data/document_parsing/final/{parser_name}/{paper_name}/{cname}"
        # )
        # out_dir.mkdir(parents=True, exist_ok=True)
        text = Path(file_path).read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        # for i, chunk in enumerate(chunks, 1):
        # out_file = out_dir / f"chunk_{i:03d}.txt"
        # with open(out_file, "w", encoding="utf-8") as f:
        #     f.write(chunk)
        print(f"✅ split into {len(chunks)} chunks")
        print(chunks[0][:500])
        return chunks


def create_vector_store_from_file(
    file_path: str,
    embedder,
    chunker_class,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> NumpyVectorStore:
    """
    Create a vector store from a file with a given embedder and chunker.

    Parameters:
    - file_path: path to the file to parse
    - embedder: an instance of BaseEmbedder (OpenAIEmbedder, E5Embedder, BGEEmbedder)
    - chunker_class: class of chunker (LengthChunker, TextStructureChunker, StructureChunker)
    - chunk_size: max chars per chunk
    - chunk_overlap: overlap chars between chunks

    Returns:
    - NumpyVectorStore object (saved to disk)
    """

    # --- 1. Load file content ---
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- 2. Chunk text ---
    chunker = chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(text)

    # --- 3. Embed chunks ---
    embeddings = embedder.embed(chunks)

    # --- 4. Build output path ---
    embedder_name = embedder.__class__.__name__
    chunker_name = chunker.__class__.__name__
    filename = Path(file_path).stem

    out_path = (
        Path("data/vector_stores")
        / embedder_name
        / str(chunk_size)
        / str(chunk_overlap)
        / chunker_name
    )
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{filename}.npz"

    # --- 5. Save to NumpyVectorStore ---
    store = NumpyVectorStore(save_path=out_file)
    store.add(embeddings, chunks)
    store.save()

    print(f"✅ Vector store saved at: {out_file}")
    return store


if __name__ == "__main__":
    path = "/Users/einsie0004/Documents/research/23_paper_extraction/pipeline/test/test_paper_sample_size_tricky.pdf"
    parsed_path = "/Users/einsie0004/Documents/research/23_paper_extraction/pipeline/data/parsed/GROBIDParser/test_paper_sample_size_tricky.md"
    # run_parsers(path)
    # create_vector_store_from_file(parsed_path, BGEEmbedder(), LengthChunker)
    embedder = BGEEmbedder()
    retriever = cross = CrossEncoderRetriever(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_m=10, k=5
    )

    query = "What is the sample size after exclusions?"
    vector_store_path = "/Users/einsie0004/Documents/research/23_paper_extraction/pipeline/data/vector_stores/BGEEmbedder/1000/100/LengthChunker/test_paper_sample_size_tricky.npz"
    store = NumpyVectorStore(save_path=vector_store_path)
    store.load()
    # Use the same embedder used for the chunks
    query_vec = embedder.embed_query(query, strategy="full")
    # Step 3: Retrieve chunks
    top_chunks = retriever.retrieve(query_vec, store, raw_query_text=query)

    print("Top-K Retrieval:")
    for txt, score in top_chunks:
        print(f"{score:.3f} :: {txt[:150]}...\n")
    # Strategy 1: full query
    # vec_full = embedder.embed_query(query, strategy="full")

    # # Strategy 2: label-definition pair
    # vec_label = embedder.embed_query(
    #     "what is the sample size after exclusions?",
    #     strategy="label",
    #     concept="Sample Size after Exclusions",
    #     definition="The number of participants which participated in the study after exclusion criteria were applied.",
    #     additional_information="Use reported numbers from the Results section",
    #     possible_answers=["N=50", "N=100", "Not reported"],
    # )
    # # Assume store is already loaded
    # results_full = store.query(vec_full, top_k=3)
    # results_label = store.query(vec_label, top_k=3)

    # print("Full query results:")
    # for txt, score in results_full:
    #     print(f"{score:.3f} :: {txt}")

    # print("\nLabel-based query results:")
    # for txt, score in results_label:
    #     print(f"{score:.3f} :: {txt}")
