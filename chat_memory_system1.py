from flask import Flask, request, jsonify
import os
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
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Process query
    tokens = [token.text for token in nlp(user_query)]
    embeddings = sentence_model.encode([user_query]).tolist()
    response = f"Default response to your query: {user_query}"

    # Log and return response
    log_interaction(user_query, response)
    return jsonify({"tokens": tokens, "embeddings": embeddings, "response": response})

# API route for fetching memory
@app.route("/memory", methods=["GET"])
def memory():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    return jsonify(memory)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
