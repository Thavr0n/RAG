# RAG
Damit das ganze funktioniert, muss erst ollama mit llama3 für die antworten und nomic-embed-text fürs embedding installiert werden. Das geht am via:

```sudo apt update```

```curl -fsSL https://ollama.com/install.sh | sh```

```ollama pull llama3```

```ollama pull nomic-embed-text```


Anschließend virtuelle python3.11 Umgebung (3.12 funktioniert nicht mit allem)
```python3.11 -m venv venv```

im entsprechenden Verzeichnis aktivieren via

```source venv/bin/activate```

Und anschließend folgende benötigte Bibliotheken installieren:

```pip3 install langchain_community```

```pip install unstructured langchain```

```pip install "unstructured[all-docs]"```



```pip install chromadb```
```pip install langchain-text-splitters```

--- 
##### unsicher ob benötigt:
```pip install psutil```

```sudo apt-get install poppler-utils```
--- 


Abschließend erst mit 
```python3.11 Dokumentenverschlinger.py ```
die ganze Datenbank auslesen, indexieren und speichern. 

Dann mit
```python3.11 FragMich.py ```
die Datenbank laden und das System antwortet nur basierend auf der Datengrundlage.
