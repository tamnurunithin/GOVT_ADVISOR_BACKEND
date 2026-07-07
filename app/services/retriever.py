from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import VECTOR_DB_DIR


def load_embeddings():
    """
    Load the same embedding model used during ingestion.
    """

    print("🔄 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded.")

    return embeddings


def load_vector_store():
    """
    Load the saved FAISS vector database.
    """

    print("\n🔄 Loading FAISS vector database...")

    embeddings = load_embeddings()

    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_DB_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ FAISS vector database loaded.")

    return vector_store


def retrieve_documents(question):
    """
    Retrieve the most relevant documents from the FAISS vector database.
    """

    print(f"\n🔍 Searching for: {question}")

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        query=question,
        k=6
    )

    print(f"✅ Retrieved {len(documents)} relevant chunks.")

    return documents


if __name__ == "__main__":

    question = "Who is eligible for the scholarship?"

    documents = retrieve_documents(question)

    print("\n========== Retrieved Chunks ==========\n")

    for i, doc in enumerate(documents, start=1):
        print(f"\n----- Chunk {i} -----\n")
        print(doc.page_content)

    print("\n======================================")