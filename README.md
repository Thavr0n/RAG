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



## Zum installieren eines Deutschen Sprachmodells für bessere Ergebnisse via Ollama:

1. git lfs installieren, um große Daten laden zu können (falls noch nicht geschehen)
2. git clone https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1
3. In den Ordner des heruntergeladenen Models gehen und via 
```docker run --rm -v .:./model ollama/quantize -q q4_K_M /model```
die daten in eine GGUF Datei umwandeln
4. Eine neue Modelfile erstellen, die auf dem llama3 Model basiert:
```# Modelfile
FROM "./Llama3-DiscoLeo-Instruct-8B-v0.1.Q4_K_M.gguf"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
TEMPLATE """
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

5. Die GGUF Datei in eine Ollama Datei um