print("1")
import os

print("2")
from langchain_community.document_loaders import PyPDFLoader

print("3")
from langchain_community.vectorstores import FAISS

print("4")
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("5")
from langchain_ollama import OllamaEmbeddings

print("6")

# -----------------------------
# Configuration
# -----------------------------
PDF_FOLDER = "./pdf"

FAISS_FOLDER = "./faiss_indexes"

print("Creating embedding object")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Embedding object created")

# -----------------------------
# Load all PDFs
# -----------------------------
documents = []

print("Loading PDFs...\n")

for file in os.listdir(PDF_FOLDER):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)

        print(f"Reading: {file}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # filename metadata add kar do
        for d in docs:
            d.metadata["source_file"] = file

        documents.extend(docs)

print(f"\nTotal pages: {len(documents)}")

# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

# -----------------------------
# Create FAISS
# -----------------------------
print("\nCreating embeddings...")

vectorstore = FAISS.from_documents(chunks, embeddings)

os.makedirs(FAISS_FOLDER, exist_ok=True)

vectorstore.save_local(FAISS_FOLDER)

print("\nKnowledge Base created successfully!")
print(f"Saved at: {FAISS_FOLDER}")