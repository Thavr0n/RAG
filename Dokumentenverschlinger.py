## ermöglicht es lokale PDFs auszulesen
from langchain_community.document_loaders import UnstructuredPDFLoader

## ermöglicht es die Texte zu Embedden und als Vektordaten zu speichern
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

## benötigt um lokale pfade um verschiedene dateien in ordnern auszulesen
import os

## gibt den Pfad zu allen Dokumenten ein
local_path = "/home/ssaman/RAG/RAG/Dokumente"

## Loop, der durch das Verzeichnis guckt, und bei der Endung .pdf aktiv wird
for filename in os.listdir(local_path):

    ## nicht verwechseln: file_path und filepath! >.<
    filepath = os.path.join(local_path, filename)
    if filename.endswith(".pdf"): ##müsste perspektivisch mit anderen dateitypen gemacht werden - dann eher als funktion?
        print(f"Processing {filename}...")

        ## der UnstructuredPDFLoader nimmt die jeweilige Datei und zerlegt sie unten weiter
        pdf_loader = UnstructuredPDFLoader(file_path=filepath)
        try:
            document = pdf_loader.load()  # Load the PDF as an unstructured document
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        ## initialisiert den TextSplitter | ChunkSize = Zusammenhängend erfasste Zeichen 
        ## Overlap = Liest auch entsprechend vor und zurück um Kontext besser zu erfassen
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)

        ## zerteilt das vom PDFLoader geladene Dokument in Chunks anhand der vom 
        #  textsplitter vorgegebenen Parameter
        chunks = text_splitter.split_documents(document)


        ## versieht jeden chunk mit einem vektor und speichert ihn ab
        ## collection_name wird gebraucht - warum auch immer
        ## eine persist funktion ist mittlerweile nichtmehr nötig
        vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag",
        persist_directory="/home/ssaman/RAG/RAG/Datenbank"
        )