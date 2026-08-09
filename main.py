from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import requests
import os

app = FastAPI()

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="pdf_documents")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Connex backend is running"}


@app.post("/ask")
def ask_question(data: Question):
    question = data.question

    print("STEP 1: Starting ChromaDB search")

    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    print("STEP 2: ChromaDB search finished")

    documents = results["documents"][0]
    context = "\n\n".join(documents)

    print("STEP 3: Sending request to Gemini")

    prompt = f"""
You are Connex, a helpful healthcare assistant that answers questions
using the provided documents.

Use ONLY the information in the documents below.

If the answer is not contained in the documents, say:
"I couldn't find that information in the provided documents."

Documents:
{context}

Question:
{question}

Answer:
"""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {"error": "GEMINI_API_KEY is not configured"}

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        },
        timeout=120
    )

    print("STEP 4: Gemini response received")

    response.raise_for_status()

    gemini_data = response.json()

    answer = gemini_data["candidates"][0]["content"]["parts"][0]["text"]

    print("STEP 5: Answer generated")

    return {
        "question": question,
        "answer": answer
    }
