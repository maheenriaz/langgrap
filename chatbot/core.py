import os

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

MD_FOLDER = "./pdf"
FAISS_FOLDER = "./faiss_indexes"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

documents = []

for file in os.listdir(MD_FOLDER):

    if not file.endswith(".md"):
        continue

    path = os.path.join(MD_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        markdown = f.read()

    # Split by headings
    header_docs = header_splitter.split_text(markdown)

    # Split long sections only
    chunks = text_splitter.split_documents(header_docs)

    for doc in chunks:

        headers = []

        if "H1" in doc.metadata:
            headers.append(doc.metadata["H1"])

        if "H2" in doc.metadata:
            headers.append(doc.metadata["H2"])

        if "H3" in doc.metadata:
            headers.append(doc.metadata["H3"])

        # Add headings to page content
        doc.page_content = (
            "\n".join(headers)
            + "\n\n"
            + doc.page_content
        )

        doc.metadata["source_file"] = file

    documents.extend(chunks)

print(f"Total chunks: {len(documents)}")

# DEBUG
for i, doc in enumerate(documents):
    print("=" * 80)
    print(doc.metadata)
    print(doc.page_content)

for doc in documents:
    if "House" in doc.page_content:
        print("helooo-----------------------")
        print("=" * 80)
        print(doc.metadata)
        print(doc.page_content)
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(FAISS_FOLDER)