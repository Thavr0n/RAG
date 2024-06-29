## benötigt für Retrieval
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# wird benötigt um auf die gespeicherten vektordaten zuzugreifen
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

persist_directory = "/home/ssaman/RAG/RAG/Datenbank"

## ermöglicht es auf die vektordatenbank, die durch Dokumentenverschlinger.py angelegt wurde zuzugreifen
emb = OllamaEmbeddings(model="nomic-embed-text")
AlleDokumente = Chroma("local-rag",persist_directory="/home/ssaman/RAG/RAG/Datenbank", embedding_function=emb)
print("Lade Vektordatenbank mit grünem Wissen...")

## legt das LLM Modell fest mit dem wir arbeiten
print("Lade LLM Model...")
try:
    local_model ="llama3:latest"
except Exception as e:
    print(f"Error: LLM model loading failed - {e}")
print("LLM Model erfolgreich geladen!")
llm = ChatOllama(model=local_model)



## Template das das LLM auffordert die fünf ähnlichsten Antwortvektoren aus der Dokumtendatenbank 
# zu dem Queryvektor aus der User*innenanfrage rauszusuchen und zu formulieren
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. 
    You answer in german only. All the documents are in german. 
    Your task is to generate five different versions 
    of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the 
    user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

## der retriever erlaubt einen String Input, was unser query samt des templates ist 
# und gibt ein Dokument als Output zurück, weshalb wir auf die in Dokumentenverschlinger.py
# angelegte Chroma Datenbank zugreifen
print("Erstelle Retriever...")
try:
    retriever = MultiQueryRetriever.from_llm(
        AlleDokumente.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,
    )
except Exception as e:
    print(f"Error: Retriever creation failed - {e}")
print("Retriever erfolgreich erstellt!")

## Dieser Befehl sagt unserem System, dass es die Frage bzw. das query NUR basierend auf den fünf besten Ergebnissen 
# unseres retrievers beantworten soll. 
template = """ Explain your answer in german based ONLY on the following context:{context}
    Question: {question}"""


prompt = ChatPromptTemplate.from_template(template)

## Chains sind eine Abfolge von Calls innerhalb von Langchain: https://js.langchain.com/v0.1/docs/modules/chains/ 
# Context gibt dem LLM weitere Hinweise, die User*innen nicht sehen, die aber bei der Bearbeitung berücksichtigt werden
# hier ist der context, der retriever mit den fünf ähnlichsten vektoren, den wir oben definiert haben
# question ist die Frage, die User*innen eingeben | RunnablePassthrough erlaubt es die Eingabeart erst später festzulegen (https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
## ebf. dazu: https://www.artefact.com/blog/unleashing-the-power-of-langchain-expression-language-lcel-from-proof-of-concept-to-production/
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Wozu soll ich Dir etwas sagen?")

## nimmt den userinput entgegen und gibt das Ergebnis aus
print(chain.invoke(input("")))



