from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router

app = FastAPI(
    title="Government Scheme Advisor API",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",
    "https://govt-advisor-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routes
app.include_router(chat_router)


@app.get("/")
def root():
    return {
        "message": "Government Scheme Advisor API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Government Scheme Advisor API is running"
    }
