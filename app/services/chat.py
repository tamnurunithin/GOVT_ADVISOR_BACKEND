from groq import Groq

from app.core.config import GROQ_API_KEY
from app.services.retriever import retrieve_documents


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def build_prompt(question, documents):
    """
    Build a prompt using the retrieved documents and user question.
    """

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are GovAssist AI, an intelligent Government Scheme Advisor.

Answer the user's question ONLY using the provided context.

If the answer is not present in the context, reply exactly with:

I couldn't find this information in the available government documents.

==========================
Context:
{context}
==========================

User Question:
{question}

Instructions:

1. Use ONLY the provided context.
2. Never make up or assume information.
3. Format the response using Markdown.
4. Use headings (##) for different sections.
5. Use bullet points instead of long paragraphs.
6. Include ONLY the sections relevant to the user's question.
7. Keep the response concise, clear, and easy to read.
8. Highlight important information using emojis.

Formatting Rules:

## Overview
Provide a short summary if applicable.

## Eligibility
Use:
✅ Requirement

Example:
✅ Resident of Telangana
✅ Minority Community
✅ Family income below ₹5 Lakhs

## Benefits
Use:
💰 Financial assistance
🎓 Scholarship support
✈️ Travel assistance

## Qualification Criteria
Use bullet points.

## Documents Required
Use:
📄 Aadhaar Card
📄 Passport
📄 Income Certificate

## Application Process
Use a numbered list.

## Countries Eligible
Use bullet points.

## Important Notes
Use:
⚠️ Important information

Do NOT include sections that are not available in the provided context.

Generate the answer now.
"""

    return prompt


def generate_answer(question):
    """
    Generate an answer using Groq and the retrieved documents.
    """

    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(question)

    # Step 2: Build the prompt
    prompt = build_prompt(question, documents)

    print("\n🔄 Sending request to Groq...\n")

    # Step 3: Call Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are GovAssist AI, a helpful Government Scheme Advisor. "
                    "Answer only using the provided context and respond in clean Markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return answer


if __name__ == "__main__":

    question = "Who is eligible for Chief Minister's Overseas Scholarship?"

    answer = generate_answer(question)

    print("\n========== AI Answer ==========\n")

    print(answer)

    print("\n===============================")