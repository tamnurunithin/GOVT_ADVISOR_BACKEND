from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import DATA_DIR, VECTOR_DB_DIR
from app.services.gemini_embeddings import GeminiEmbeddings


def load_documents():
    """
    Load all PDF documents from the data directory.
    """

    documents = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in the data folder.")
        return []

    print(f"\n📚 Found {len(pdf_files)} PDF(s)\n")

    for pdf in pdf_files:
        print(f"📄 Loading: {pdf.name}")

        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        # Store filename in metadata
        for doc in docs:
            doc.metadata["source_file"] = pdf.name

        documents.extend(docs)

    print(f"\n✅ Total Pages Loaded: {len(documents)}")

    return documents


def split_documents(documents):
    """
    Split PDF pages into chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\n✅ Total Chunks Created: {len(chunks)}")

    return chunks


def create_embeddings():
    """
    Load Gemini Embeddings.
    """

    print("\n🔄 Loading Gemini Embedding model...")

    embeddings = GeminiEmbeddings()

    print("✅ Gemini Embedding model ready.")

    return embeddings


def create_vector_store(chunks, embeddings):
    """
    Create and save the FAISS vector database.
    """

    print("\n🔄 Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    vector_store.save_local(str(VECTOR_DB_DIR))

    print(f"✅ FAISS vector database saved to:\n{VECTOR_DB_DIR}")

    return vector_store


if __name__ == "__main__":

    print("\n========== AI Government Scheme Advisor ==========\n")

    documents = load_documents()

    if not documents:
        exit()

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\n========== First Chunk ==========\n")

    print(chunks[0].page_content)

    print("\n=================================\n")