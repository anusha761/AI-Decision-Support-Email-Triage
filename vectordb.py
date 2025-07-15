from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

# Initialize embeddings model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # Keep paragraph separation
    return text.strip()



def prepare_chunks_with_metadata(text, department):
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

