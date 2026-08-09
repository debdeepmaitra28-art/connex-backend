from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Where our extracted text files are
TEXT_FOLDER = Path(".")

# Where ChromaDB will save its database
DB_FOLDER = "chroma_db"

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create/open the ChromaDB database
client = chromadb.PersistentClient(path=DB_FOLDER)

# Create a collection for our Connex documents
collection = client.get_or_create_collection(name="pdf_documents")


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# Find all extracted text files
text_files = list(TEXT_FOLDER.glob("*.txt"))

print(f"Found {len(text_files)} text files.")

all_chunks = []
all_ids = []
all_metadata = []

chunk_number = 0

for file_path in text_files:
    print(f"Processing: {file_path.name}")

    text = file_path.read_text(encoding="utf-8")

    chunks = split_text(text)

    for chunk in chunks:
        if chunk.strip():
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_number}")
            all_metadata.append({
                "source": file_path.name
            })
            chunk_number += 1


print(f"Created {len(all_chunks)} chunks.")

# Create embeddings and store everything in ChromaDB
collection.add(
    ids=all_ids,
    documents=all_chunks,
    metadatas=all_metadata,
)

print("RAG database created successfully!")
