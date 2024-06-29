## ermöglicht es lokale PDFs auszulesen
from langchain_community.document_loaders import UnstructuredPDFLoader

## ermöglicht es die Texte zu Embedden und als Vektordaten zu speichern
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

## benötigt um lokale pfade und dateien in ordnern auszulesen
import os

## gibt den Pfad zu allen Dokumenten ein
local_path = "/home/ssaman/SimpleRAG/SimpleRAG/venv/Dokumente"

## initialisiert den TextSplitter | ChunkSize = Zusammenhängend erfasste Zeichen 
## Overlap = Liest auch entsprechend vor und zurück um Kontext zu erfassen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=50)

vector_db = None   

def index_pdf(data):
    global vector_db
    try:
        # Split the text into chunks using the RecursiveCharacterTextSplitter
        chunks = text_splitter.split_documents(data) 

        # Initialize the Chroma database if it hasn't been initialized 
        
        for chunk in chunks:
            # Embed the chunk using OllamaEmbeddings and save it to the Chroma database
            vector_db = Chroma.from_documents(
                documents=chunk, 
                embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
                persist_directory="/home/ssaman/SimpleRAG/SimpleRAG/venv/Datenbank"
            )
    except Exception as e:
        print(f"Error loading document: {data} - {str(e)}")


for file in os.listdir(local_path):
    if file.endswith(".pdf"):
        loader = UnstructuredPDFLoader(local_path)
        data = loader.load()
        ## versichert, dass erfolgreich geladen wurde
        if not data or len(data) == 0:
            print(f"Skipped document: {file} (no text found)")
            continue
        print(f"Loaded coument:{file.split('.')[0]}")
        index_pdf(data)

