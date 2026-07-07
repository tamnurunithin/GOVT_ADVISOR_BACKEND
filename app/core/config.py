from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Backend root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)