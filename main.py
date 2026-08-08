from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import requests

app = FastAPI()

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="pdf_documents")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Connex backend is running"}


@app.post("/ask")
def ask_question(data: Question):
    question = data.question

    print("STEP 1: Starting ChromaDB search")

    # Search ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    print("STEP 2: ChromaDB search finished")

    # Get documents
    documents = results["documents"][0]

    # Build context
    context = "\n\n".join(documents)

    print("STEP 3: Sending request to Ollama")

    # Prompt for Ollama
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

    # Send request to Ollama
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    print("STEP 4: Ollama response received")

    response.raise_for_status()

    answer = response.json()["response"]

    print("STEP 5: Answer generated")

    return {
        "question": question,
        "answer": answer
    }