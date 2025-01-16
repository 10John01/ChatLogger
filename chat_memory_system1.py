import os
from flask import Flask, request, jsonify
import json
import spacy
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load models
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Memory log file
MEMORY_FILE = "chat_memory.json"

# Initialize memory
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

# Function to log interactions
def log_interaction(query, response):
    with open(MEMORY_FILE, "r+") as f:
        memory = json.load(f)
        memory.append({"query": query, "response": response})
        f.seek(0)
        json.dump(memory, f, indent=4)

# API route for querying
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        user_query = data["query"]
        tokens = [token.text for token in nlp(user_query)]
        embeddings = sentence_model.encode([user_query]).tolist()
        response = f"Default response to your query: {user_query}"

        log_interaction(user_query, response)
        return jsonify({"tokens": tokens, "embeddings": embeddings, "response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API route for fetching memory
@app.route("/memory", methods=["GET"])
def memory():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    return jsonify(memory)

if __name__ == "__main__":
    # Dynamically determine the port Render assigns or default to 5000
    port = int(os.environ.get("PORT", 22531))  # Default port is 22531 (Extra Cool)
    if port < 1024 or (7000 <= port <= 9000) or port > 49151:
        raise ValueError(f"Invalid port {port}: must be 1024–49151 excluding 7000–9000.")
    app.run(host="0.0.0.0", port=port, debug=True)
