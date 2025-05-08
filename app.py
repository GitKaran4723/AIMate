from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import sqlite3
import os
from dotenv import load_dotenv
import pytz

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from google import genai
from google.genai import types

app = Flask(__name__)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Timezone
IST = pytz.timezone("Asia/Kolkata")
def get_india_time():

    print(datetime.now(IST))
    return datetime.now(IST)

# Chroma setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client(Settings())
profile_mem = chroma_client.get_or_create_collection("profile_memory")
event_mem = chroma_client.get_or_create_collection("event_memory")

# SQLite setup
def init_db():
    conn = sqlite3.connect("assistant.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        timestamp TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS diary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        reflection TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        message TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

def save_conversation(role, message, timestamp):
    conn = sqlite3.connect("assistant.db")
    conn.execute("INSERT INTO conversation_log (role, message, timestamp) VALUES (?, ?, ?)", (role, message, timestamp))
    conn.commit()
    conn.close()

def load_recent_conversation(limit=50, offset=0):
    db_path = "assistant.db"
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'")
        if cursor.fetchone() is None:
            conn.close()
            return []

        rows = cursor.execute("""
            SELECT role, message, timestamp FROM conversation_log
            ORDER BY id DESC LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
        conn.close()

        return list(reversed([{"role": row[0], "text": row[1], "timestamp": row[2]} for row in rows]))

    except Exception as e:
        print(f"[ERROR] Failed to load conversation: {e}")
        return []

def embed(text):
    return embedding_model.encode([text])[0].tolist()

def store_profile_fact(fact, fact_id):
    profile_mem.add(documents=[fact], embeddings=[embed(fact)], ids=[fact_id])

def query_profile(question, k=2):
    return profile_mem.query(query_embeddings=[embed(question)], n_results=k)["documents"][0]

def store_event(event):
    now = get_india_time().isoformat()
    conn = sqlite3.connect("assistant.db")
    conn.execute("INSERT INTO events (text, timestamp) VALUES (?, ?)", (event, now))
    conn.commit()
    conn.close()
    event_mem.add(documents=[event], embeddings=[embed(event)], ids=[now])

def query_events(query, k=3):
    return event_mem.query(query_embeddings=[embed(query)], n_results=k)["documents"][0]

def store_daily_reflection(text):
    date = get_india_time().strftime("%Y-%m-%d")
    conn = sqlite3.connect("assistant.db")
    conn.execute("INSERT INTO diary (date, reflection) VALUES (?, ?)", (date, text))
    conn.commit()
    conn.close()

def fetch_diary():
    conn = sqlite3.connect("assistant.db")
    rows = conn.execute("SELECT date, reflection FROM diary ORDER BY date DESC").fetchall()
    conn.close()
    return rows

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message").strip()
    now = get_india_time()
    timestamp = now.isoformat()

    recent_history = load_recent_conversation(limit=50)

    system_instruction = {
        "role": "user",
        "text": (
            "You are Chinni, a caring and supportive AI assistant created to help Karan achieve his dreams. "
            "Respond naturally like a real close friend. Use warm, gentle language. Keep your responses small and kind. "
            "You can split longer thoughts into multiple messages and continue if interrupted. Do not reference previous chats unless asked explicitly."
        )
    }

    contents = [
        types.Content(role=system_instruction["role"], parts=[types.Part(text=system_instruction["text"])])
    ] + [
        types.Content(role=msg["role"], parts=[types.Part(text=msg["text"])])
        for msg in recent_history
    ] + [
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    now_dt = get_india_time()
    time_context = f"The current Indian time is {now_dt.strftime('%I:%M %p')}, and the date is {now_dt.strftime('%A, %d %B %Y')} (IST)."
    contents.insert(1, types.Content(role="user", parts=[types.Part(text=time_context)]))

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.GenerateContentConfig(response_mime_type="text/plain")

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=config
        ):
            response_text += chunk.text

        save_conversation("user", user_input, timestamp)
        save_conversation("model", response_text.strip(), timestamp)

        return jsonify({"response": response_text.strip(), "timestamp": timestamp})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "timestamp": timestamp})

@app.route('/history', methods=['GET'])
def history():
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))
    messages = load_recent_conversation(limit=limit, offset=offset)
    return jsonify(messages)

@app.route('/')
def index():
    conn = sqlite3.connect("assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversation_log")
    total_messages = cursor.fetchone()[0]
    conn.close()

    if total_messages > 50:
        chat_history = load_recent_conversation(limit=50)
    else:
        chat_history = load_recent_conversation(limit=total_messages)

    return render_template("chat.html", chat_history=chat_history, now=get_india_time())

@app.route('/profile', methods=['POST'])
def update_profile():
    data = request.json
    fact = data["fact"]
    fact_id = str(get_india_time().timestamp())
    store_profile_fact(fact, fact_id)
    return jsonify({"status": "stored", "fact": fact})

@app.route('/event', methods=['POST'])
def log_event():
    event = request.json["event"]
    store_event(event)
    return jsonify({"status": "logged", "event": event})

@app.route('/diary', methods=['POST'])
def log_diary():
    reflection = request.json["reflection"]
    store_daily_reflection(reflection)
    return jsonify({"status": "saved", "reflection": reflection})

@app.route('/time')
def get_time():
    now = get_india_time()
    return jsonify({
        "datetime": now.isoformat(),
        "date": now.strftime("%A, %d %B %Y"),
        "time": now.strftime("%I:%M %p"),
    })

@app.template_filter('todatetime')
def to_datetime_filter(value):
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(IST)  # fallback to current time


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)