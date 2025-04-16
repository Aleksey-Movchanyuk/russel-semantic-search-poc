import json
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer


# ---- CONFIG ----
JSONL_FILE = "/Users/anymacstore/Downloads/text.data.jsonl"
COLLECTION_NAME = "cap_cases"
BATCH_SIZE = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ID_TRACK_FILE = "uploaded_ids.txt"


# ---- INITIALIZE ----
model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host="localhost", port=6333)


def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def load_uploaded_ids() -> set:
    if Path(ID_TRACK_FILE).exists():
        with open(ID_TRACK_FILE, "r") as f:
            return set(map(int, f.read().splitlines()))
    return set()


def save_uploaded_ids(ids):
    with open(ID_TRACK_FILE, "a") as f:
        for _id in ids:
            f.write(f"{_id}\n")


def stream_cases(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text_for_embedding(case: dict) -> str:
    try:
        return case["casebody"]["data"]["opinions"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def upload_cases(filepath: str, batch_size: int = 100):
    total_lines = count_lines(filepath)
    uploaded_ids = load_uploaded_ids()
    new_uploaded_ids = []
    points = []

    processed = 0
    last_percent = -1

    for case in stream_cases(filepath):
        case_id = int(case["id"])
        processed += 1
        percent = int((processed / total_lines) * 100)

        if percent != last_percent:
            print(f"✅ {percent}% uploaded...")
            last_percent = percent

        if case_id in uploaded_ids:
            continue

        text = extract_text_for_embedding(case)
        if not text.strip():
            continue

        try:
            vector = model.encode(text)
        except Exception as e:
            print(f"⚠️ Skipping case ID {case_id}: {e}")
            continue

        points.append({
            "id": case_id,
            "vector": vector,
            "payload": case  # 🔥 full record goes here
        })

        new_uploaded_ids.append(case_id)

        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            save_uploaded_ids(new_uploaded_ids)
            uploaded_ids.update(new_uploaded_ids)
            new_uploaded_ids = []
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        save_uploaded_ids(new_uploaded_ids)

    print("✅ Upload complete!")


if __name__ == "__main__":
    if not Path(JSONL_FILE).exists():
        print(f"❌ File not found: {JSONL_FILE}")
    else:
        create_collection()
        upload_cases(JSONL_FILE, batch_size=BATCH_SIZE)
