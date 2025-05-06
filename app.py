from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import sqlite3
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

app = Flask(__name__)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def init_db():
    conn = sqlite3.connect('chatmate.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            ai_response TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def fetch_full_conversation():
    conn = sqlite3.connect('chatmate.db')
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, ai_response, timestamp FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/')
def index():
    # Load past chat for the frontend
    past_chats = fetch_full_conversation()
    now = datetime.now()
    return render_template('chat.html', chat_history=past_chats, now=now, timedelta=timedelta)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    timestamp = datetime.now().isoformat()

    # Use full conversation history as context
    past = fetch_full_conversation()
    context = "".join([f"User: {u}\nAI: {a}\n" for u, a, t in past])
    full_prompt = context + f"User: {user_input}\nAI:"

    ai_response = gemini_ai_response(full_prompt)

    conn = sqlite3.connect('chatmate.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (user_message, ai_response, timestamp)
        VALUES (?, ?, ?)
    """, (user_input, ai_response, timestamp))
    conn.commit()
    conn.close()

    return jsonify({"response": ai_response, "timestamp": timestamp})

def gemini_ai_response(prompt):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-2.5-flash-preview-04-17"
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        return response_text.strip()
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"

@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect('chatmate.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([
        {"id": r[0], "user_message": r[1], "ai_response": r[2], "timestamp": r[3]} for r in rows
    ])

@app.template_filter('todatetime')
def to_datetime_filter(value):
    return datetime.fromisoformat(value)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)