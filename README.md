# 🧠 Caselaw Dataset Semantic Search (Qdrant + SentenceTransformers)

This mini-project provides a semantic search engine for the [Caselaw Access Project (CAP)](https://case.law/) dataset using **Qdrant** as a vector store and **SentenceTransformers** to generate embeddings.

## 🚀 Project Structure

```
.
├── cap_cases_query_sample.py        # Example script to run semantic queries
├── cap_cases_dataset_uploader.py  # Uploads CAP .jsonl data to Qdrant with embeddings
├── docker-compose.yml             # Launches local Qdrant instance
├── uploaded_ids.txt               # Track already uploaded case IDs
├── requirement.txt                # Python dependencies
├── README.md                      # This file
└── .gitignore
```

---

## 🏁 Quickstart

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/Aleksey-Movchanyuk/russel-semantic-search-poc.git
cd russel-semantic-search-poc
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

### 2. Start Qdrant (Local Vector DB)

```bash
docker-compose up -d
```

Qdrant will be available at: [http://localhost:6333](http://localhost:6333)

---

## 📦 Upload Dataset

Download a subset of the Caselaw Access Project in `.jsonl` format and run:

```bash
python cap_cases_dataset_uploader.py
```

- Embeddings are generated using `all-MiniLM-L6-v2` via `sentence-transformers`.
- Each case is stored as a vector with full metadata and opinion text.
- Progress and status are logged.
- Uploaded IDs are tracked in `uploaded_ids.txt` to avoid duplicates.

---

## 🔍 Run Semantic Query

Edit your query inside `cap_cases_query_sample.py` and run:

```bash
python cap_cases_query_sample.py
```

Output:

```
📚 Score: 0.824
🧾 Case: People v. Smith (2005-06-23)
📄 Snippet:
The defendant entered the premises unlawfully with a clear intent to commit theft, based on...
```

---

## 🧠 Model Used

- Embedding Model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Vector Dimension: 384
- Search Backend: Qdrant (Cosine Similarity)

---

## 📄 Example Query

```python
query = "unlawful entry and intent to commit a crime"
```

---

## 📎 Notes

- You can test with any CAP `.jsonl` subset from Hugging Face or [case.law](https://case.law).
- Supports incremental upload and repeatable querying.

---

## 📜 License

MIT — free to use for academic or personal projects.

---

## 🙏 Acknowledgements

- [Caselaw Access Project](https://case.law/)
- [Qdrant](https://qdrant.tech)
- [SentenceTransformers](https://www.sbert.net/)
