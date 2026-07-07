from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import generate_answer

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):
    """
    Chat endpoint for the AI Government Scheme Advisor.
    """

    answer = generate_answer(request.question)

    return ChatResponse(answer=answer)