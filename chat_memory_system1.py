from flask import Flask, request, jsonify
import os
import json
import spacy
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
MEMORY_FILE = "chat_memory.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

def log_interaction(query, response):
    with open(MEMORY_FILE, "r+") as f:
        memory = json.load(f)
        memory.append({"query": query, "response": response})
        f.seek(0)
        json.dump(memory, f, indent=4)

@app.route("/query", methods=["POST"])
def query():
    try:
        # Log raw request data
        print(f"Raw Data: {request.data}")  # Add this line to log incoming data
        print(f"Headers: {request.headers}")  # Log headers to ensure correct Content-Type

        # Parse JSON payload
        data = request.json
        if not data or "query" not in data:
            print("Error: No query provided")
            return jsonify({"error": "No query provided"}), 400

        user_query = data["query"]
        print(f"User query: {user_query}")

        tokens = [token.text for token in nlp(user_query)]
        embeddings = sentence_model.encode([user_query]).tolist()
        response = f"Default response to your query: {user_query}"

        log_interaction(user_query, response)
        return jsonify({"tokens": tokens, "embeddings": embeddings, "response": response})

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/memory", methods=["GET"])
def memory():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    return jsonify(memory)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=55515, debug=True)
