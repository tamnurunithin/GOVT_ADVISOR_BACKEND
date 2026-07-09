from google import genai
from langchain_core.embeddings import Embeddings

from app.core.config import GEMINI_API_KEY


class GeminiEmbeddings(Embeddings):
    """
    Gemini Embeddings implementation compatible with LangChain FAISS.
    """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemini-embedding-001"

    def embed_documents(self, texts):
        embeddings = []

        print(f"🔄 Creating embeddings for {len(texts)} chunks...")

        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )

            embeddings.append(response.embeddings[0].values)

        print("✅ Document embeddings created.")

        return embeddings

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return response.embeddings[0].values