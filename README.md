# 🧠 chonkychunker

**chonkychunker** is a lightweight and customizable Python library for semantically chunking and clustering texts using `SentenceTransformers` and `BallTree` or `KNN`. It’s ideal for preparing grouped content for vector databases, semantic search systems, or integration into LangChain-based RAG pipelines.

---

## 🚀 Features

- ✨ Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`)
- 🧠 Clusters similar texts using Ball Tree or Nearest Neighbors (KNN)
- 🔁 Deduplicates overlapping clusters
- 🔗 Outputs clusters as:
  - List of grouped text
  - LangChain-compatible `Document` objects
  - Vector DB-friendly dicts with embeddings
- 📎 `merge=True` support: Combine all texts in a cluster
- 🎯 `max_tokens=`: Truncate merged content for context limit safety

---

## 📦 Installation

```bash
pip install chonkychunker
```

Or from source:

```bash
git clone https://github.com/aravindraju98/chonkychunker.git
cd chonkychunker
pip install -e .
```

---


## 🔧 Constructor Arguments

```python
TextChunker(
    metric='euclidean',     # 'euclidean' (default, uses BallTree) or 'cosine' (uses KNN)
    top_k=5,                # Number of nearest neighbors per point
    distance_threshold=2,   # Distance threshold for inclusion in cluster
    max_tokens=None         # Optional token cap on merged text
)
```

## 🧪 Quickstart Example

```python
from chonkychunker import TextChunker

texts = [
    "The milk is spoiled.",
    "Eggs are boiled and tasty.",
    "Physics involves matter and energy.",
    "Salt is added for flavor.",
    "Thermonuclear reactions are powerful."
]

chunker = TextChunker(top_k=3, max_tokens=50) #using default metric='euclidean', distance_threshold = 2
chunker.embed(texts)

# Vector output with merged clusters
vector_data = chunker.get_vector_output(merge=True)

# LangChain Documents (merged)
docs = chunker.to_langchain_documents(merge=True)

#content of vector_data
[{'id': 'cluster_0',
 'text': 'The milk is spoiled.\nEggs are boiled and tasty.\nSalt is added for flavor.',
 'embedding': [-0.005473766475915909..,...0.0836096704006195,  -0.02507365308701992],
 'cluster': 0},
{'id': 'cluster_1',
  'text': 'Physics involves matter and energy.\nThermonuclear reactions are powerful.',
  'embedding': [-0.024956505745649338,.....0.0836096704006195,  -0.02507365308701992],
  'cluster': 0}]

#content of docs

[Document(metadata={'cluster': 0}, page_content='The milk is spoiled.\nEggs are boiled and tasty.\nSalt is added for flavor.'),
 Document(metadata={'cluster': 1}, page_content='Physics involves matter and energy.\nThermonuclear reactions are powerful.')]


```



---

## 🔄 Merge Option

Use `merge=True` in:
- `get_vector_output(merge=True)`
- `to_langchain_documents(merge=True)`

This will concatenate all texts in a cluster into one document. If `max_tokens` is set, it will truncate the combined text based on token count using the Sentence-BERT tokenizer.

---

## 🧠 Cosine Distance vs Euclidean

- Default distance metric: `euclidean` (used with `BallTree`)
- Set `metric='cosine'` to switch to `NearestNeighbors` (KNN)
  ```python
  TextChunker(metric='cosine', ...)
  ```

---

## 📘 LangChain Integration

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)
```
---
## JSON Export

You can export clustered text and embedding data as JSON — either saved to disk or returned as a JSON string.

🔹 Save to File:
```python

chunker = TextChunker(top_k=3, max_tokens=50) #using default metric='euclidean', distance_threshold = 2
chunker.embed(texts)
chunker.to_json(merge=True, filepath="output.json")
 ```

🔹 Return the data in json format
```python

chunker = TextChunker(top_k=3, max_tokens=50) #using default metric='euclidean', distance_threshold = 2
chunker.embed(texts)
json_str = chunker.to_json(merge=True, return_data=True)

```


🔧 Parameters:

    merge (bool): Whether to merge cluster texts

    filepath (str): Output file path (if not using return_data)

    return_data (bool): If True, returns a JSON string instead of saving
---

## 📜 License

MIT License © 2024 [Aravind Raju](https://github.com/aravindraju98)
