## ermöglicht es lokale PDFs auszulesen
from langchain_community.document_loaders import UnstructuredPDFLoader

## ermöglicht es die Texte zu Embedden und als Vektordaten zu speichern
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

## benötigt um lokale pfade und dateien in ordnern auszulesen
import os

## gibt den Pfad zu allen Dokumenten ein
local_path = "/home/ssaman/RAG/RAG/Dokumente"

for filename in os.listdir(local_path):
    filepath = os.path.join(local_path, filename)
    if filename.endswith(".pdf"):  # Check if the file is a PDF
        print(f"Processing {filename}...")
        pdf_loader = UnstructuredPDFLoader(file_path=filepath)
        try:
            document = pdf_loader.load()  # Load the PDF as an unstructured document
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        ## initialisiert den TextSplitter | ChunkSize = Zusammenhängend erfasste Zeichen 
        ## Overlap = Liest auch entsprechend vor und zurück um Kontext zu erfassen
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8)
        chunks = text_splitter.split_documents(document)

        vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag",
        persist_directory="/home/ssaman/RAG/RAG/Datenbank"
        )