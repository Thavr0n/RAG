# RAG - Wozu dient dieser Code?
Dieses Setup dient dazu ein loakles Retrieval Augmented Generation (RAG) System mit einer lokalen LLM zu verknüpfen. Die Datei Dokumentenverschlinger.py ermöglicht es beliebig viel Text aus PDFs auszulesen, zu embedden und in einer Vektordatenbank zu speichern. Ausführen mit:

``` python3.11 Dokumentenverschlinger.py```

Anschließend wird die Datenbank von FragMich.py geladen und beantwortet einen User*innen-Input anhand der Datenbank. Dies geschieht mit Ein- und Ausgabe im Terminal. Ausführen mit:

``` python3.11 FragMich.py ``` 

Der Code ist ausführlichst auskommentiert, da er mir selbst beim Erlernen und Verstehen des Systems half. Anderen ebf. unerfahrenen Programmierenden kann er evtl. dabei helfen das Framework schneller zu erfassen.


## Erste Schritte bei der eigenen Umsetzung:
#### Installieren von Ollama ("Docker für LLMs")
Damit das ganze funktioniert, muss erst ollama mit llama3 für die antworten und nomic-embed-text fürs embedding installiert werden. Das geht via:

```sudo apt update```

```curl -fsSL https://ollama.com/install.sh | sh```

```ollama pull llama3```

```ollama pull nomic-embed-text```

#### Starten einer virtuellen Umgebung
Anschließend virtuelle python3.11 Umgebung (3.12 funktioniert nicht mit allem)
```python3.11 -m venv venv```

im entsprechenden Verzeichnis aktivieren via

```source venv/bin/activate```

#### Installieren notwendiger Bibliotheken
Und daraufhin folgende benötigte Bibliotheken installieren:

```pip3 install langchain_community```

```pip install unstructured langchain```

```pip install "unstructured[all-docs]"```

```pip install chromadb```

```pip install langchain-text-splitters```

#### Erstellung der Datenbank und Query an die Datenbank
Abschließend erst mit 
```python3.11 Dokumentenverschlinger.py ```
die ganze Datenbank auslesen, indexieren und speichern. 

Dann mit
```python3.11 FragMich.py ```
die Datenbank laden und das System antwortet nur basierend auf der Datengrundlage.



## Installieren eines Deutschen Sprachmodells für bessere Ergebnisse 

Diese Schritte sind nur notwendig, wenn das gewünschte Sprachmodell noch nicht in der ollama library verfügbar ist und nur als safetensor bzw. gguf modell existiert:
Ebenfalls sind sie nur möglich, wenn das basierende Sprachmodell auf einer dieser drei Architekturen beruht (vgl. https://github.com/ollama/ollama/blob/main/docs/import.md#importing-pytorch--safetensors )
- LlamaForCausalLM
- MistralForCausalLM
- GemmaForCausalLM 

Ob das so ist kann in der Repo des Modells unter config.json eingesehen werden. 

Nachdem die obigen Voraussetzungen überprüft sind:
1. git lfs installieren, um große Daten laden zu können (falls noch nicht geschehen)
2. ```git clone https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1```

4. Eine neue Modelfile erstellen, die als Quelle den eben heruntergeladenen Ordner angibt:
``` ## Modelfile ``` 

``` FROM /home/ssaman/Llama3-DiscoLeo-Instruct-8B-v0.1"```

 ``` PARAMETER stop "<|im_start|>"``` 

``` PARAMETER stop "<|im_end|>"``` 

``` TEMPLATE """``` 

``` <|im_start|>system``` 

``` {{ .System }}<|im_end|>``` 

``` <|im_start|>user``` 

``` {{ .Prompt }}<|im_end|>``` 

``` <|im_start|>assistant``` 

``` """``` 


5. Den Ordnerpfad in dem Safetensors/GGUF-Daten und Modelfile liegen öffnen und mit Ollama eine neue LLM daraus erstellen via:
```ollama create -q Q4_K_M GermanLlama3```
Achtung dieser Schritt erfordert viel freien Speicherplatz (Bei mir bis zu 60GB | Das fertiggestellte Model hat dann allerdings nur 4,9GB)
Weitere Quantisierungsmethoden ("Q4_K_M") hier: https://github.com/ollama/ollama/blob/main/docs/import.md#importing-pytorch--safetensors 


## Installieren von Pipelines um das System mit OpenWebUI zu verbinden
https://github.com/open-webui/pipelines 


1. Installieren von OpenWebUI:
```docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main```

2. Installieren von Pipelines mit erweiterten Dependencies:
```docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -e PIPELINES_URLS="https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py" -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main```

3. Über Einstellungen --> Admin Panel --> Connections die Verbindung zu Pipelines und Ollama überprüfen mit einem Klick auf die zwei Pfeile rechts neben den Eingabefeldern. 
Open AI API: ```http://localhost:9099``` API KEY: ```0p3n-w3bu!```
Ollama API: ```http://127.0.0.1:11434```

---- falls die Verbindung mit Pipelines nicht funktioniert. Docker container von OpenWebUI stoppen, löschen und neu starten mit 
```docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main``` 
