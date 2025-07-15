!pip install langchain chromadb sentence-transformers
!pip install -U langchain-community
!pip install PyPDF2

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize embeddings model (offline, local)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text


# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Extract text from your PDF files
# finance_sop_text = extract_text_from_pdf("Finance_SOP.pdf")
# it_sop_text = extract_text_from_pdf("IT_SOP.pdf")

# # Setup splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separators=[" \n \n","\n", " ", ""]
# )

# # Split the Finance SOP
# finance_chunks = text_splitter.split_text(finance_sop_text)

# # Split the IT SOP
# it_chunks = text_splitter.split_text(it_sop_text)

# # Just printing number of chunks as example
# print(f"Finance SOP chunks: {len(finance_chunks)}")
# print(f"IT SOP chunks: {len(it_chunks)}")


import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # Keep paragraph separation
    return text.strip()

def prepare_chunks_with_metadata(text: str, department: str):
    """Split text into chunks and attach metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[" \n \n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    # Prepare list of dicts for each chunk with metadata
    chunked_data = []
    for chunk in chunks:
        chunked_data.append({
            "text": chunk,
            "metadata": {"department": department}
        })
    return chunked_data

def main():
    # Extract text from PDFs
    finance_text = extract_text_from_pdf("Finance_SOP.pdf")
    it_text = extract_text_from_pdf("IT_SOP.pdf")

    # Prepare chunks with metadata
    finance_chunks = prepare_chunks_with_metadata(finance_text, "Finance/Tax")
    it_chunks = prepare_chunks_with_metadata(it_text, "IT")

    # Combine all chunks for one unified vector store ingestion
    all_chunks = finance_chunks + it_chunks

    print(f"Total chunks to ingest: {len(all_chunks)}")
    # Example print of first chunk and metadata
    print("Sample chunk with metadata:")
    print(all_chunks[0]["text"])
    print(all_chunks[0]["metadata"])

    # ingest chunks
    # Extract texts and metadatas separately from all_chunks
    texts = [item["text"] for item in all_chunks]
    metadatas = [item["metadata"] for item in all_chunks]

    # Create or load Chroma vector store
    chroma_collection = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        persist_directory="chroma_sop",
        collection_name="sop_collection"
    )

    # Persist to disk
    chroma_collection.persist()

    print(f"Ingested {len(texts)} chunks into ChromaDB.")

if __name__ == "__main__":
    main()



!zip -r chroma_sop.zip chroma_sop
