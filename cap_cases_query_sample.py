from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
COLLECTION_NAME = "cap_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- INIT ----
model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host="localhost", port=6333)

# ---- QUERY ----
query = "unlawful entry and intent to commit a crime"
query_vector = model.encode(query).tolist()

# ---- SEARCH ----
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,
    with_payload=True
)

# ---- PRINT RESULTS ----
for hit in results:
    payload = hit.payload
    try:
        case_name = payload.get("name_abbreviation", payload.get("name"))
        case_date = payload.get("decision_date", "Unknown date")
        opinion_text = payload["casebody"]["data"]["opinions"][0]["text"]
        snippet = opinion_text.strip().replace("\n", " ")[:300]
        print(f"\n📚 Score: {hit.score:.3f}")
        print(f"🧾 Case: {case_name} ({case_date})")
        print(f"📄 Snippet:\n{snippet}...\n")
    except Exception as e:
        print(f"⚠️ Could not parse result: {e}")
