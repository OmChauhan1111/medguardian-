# create_collections.py
import os
import json
import bcrypt
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING

# Load from environment or Streamlit secrets fallback
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME") or "medguardian"

if not MONGO_URI:
    raise SystemExit("MONGO_URI not set. Set environment variable or add to .streamlit/secrets.toml")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

# Names of collections
USERS = "users"
REPORTS = "reports"
CHATS = "chats"

def create_collections_and_indexes():
    # Create (get) collections (Mongo creates on first insert, but we can ensure and create indexes)
    users_col = db[USERS]
    reports_col = db[REPORTS]
    chats_col = db[CHATS]

    print("Creating indexes...")
    # Unique username
    users_col.create_index([("username", ASCENDING)], unique=True)
    # Reports query by user_id, recent first
    reports_col.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    # Chats for ordering
    chats_col.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
    print("Indexes created.")

def insert_sample_documents():
    users_col = db[USERS]
    reports_col = db[REPORTS]
    chats_col = db[CHATS]

    # Sample user (password hashed with bcrypt)
    sample_user = {
        "username": "sample_user",
        "password_hash": bcrypt.hashpw("sample_pass".encode(), bcrypt.gensalt()).decode(),
        "full_name": "Sample User",
        "phone": "+91 99999 00000",
        "created_at": datetime.utcnow()
    }
    try:
        ures = users_col.insert_one(sample_user)
        print("Inserted sample user id:", str(ures.inserted_id))
        user_id = str(ures.inserted_id)
    except Exception as e:
        print("Sample user insert failed (maybe exists):", e)
        found = users_col.find_one({"username":"sample_user"})
        user_id = str(found["_id"]) if found else None

    # Sample report (structure same as app expects)
    sample_report = {
        "user_id": user_id,
        "patient_id": "MG-TEST-001",
        "patient_name": "Test Patient",
        "phone": "+91 88888 11111",
        "doctor_name": "Dr. Test",
        "referred_by": "Self",
        "sample_collected": datetime.utcnow().strftime("%d-%m-%Y"),
        "report_generated_by": "MedGuardian Test",
        "date": datetime.utcnow().strftime("%d-%m-%Y %I:%M %p"),
        "condition_name": "Heart",
        "risk": 23.5,
        "raw_json": json.dumps({"note":"sample report"}, default=str),
        "created_at": datetime.utcnow()
    }

    rres = reports_col.insert_one(sample_report)
    print("Inserted sample report id:", str(rres.inserted_id))

    # Sample chat
    chat1 = {"user_id": user_id, "role": "user", "message": "Hello Doctor", "created_at": datetime.utcnow()}
    chat2 = {"user_id": user_id, "role": "bot", "message": "Hello, how can I help?", "created_at": datetime.utcnow()}
    chats_col.insert_many([chat1, chat2])
    print("Inserted sample chat messages.")

def list_collections_and_counts():
    print("Collections in DB:", db.list_collection_names())
    for name in [USERS, REPORTS, CHATS]:
        c = db[name]
        print(f"{name} count:", c.count_documents({}))

if __name__ == "__main__":
    create_collections_and_indexes()
    insert_sample_documents()
    list_collections_and_counts()
    print("Done.")
