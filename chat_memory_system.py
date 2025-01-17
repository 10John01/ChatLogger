import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
import spacy
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client  # Supabase client library

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Load from environment variable
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Load from environment variable

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flask app
app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to log interactions in Supabase
def log_interaction(query, response, interaction_type="text"):
    try:
        entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "keywords": [token.text for token in nlp(query) if token.is_alpha],
            "embeddings": sentence_model.encode([query]).tolist(),
            "type": interaction_type,
        }
        # Insert into Supabase
        result = supabase.table("memory_entries").insert(entry).execute()
        if result.status_code != 200:
            print(f"Error logging interaction: {result.json()}")
    except Exception as e:
        print(f"Error logging interaction: {e}")

# Search memory in Supabase
def search_memory(query, max_results=5):
    try:
        embeddings = sentence_model.encode([query]).tolist()
        # Fetch all memory entries
        response = supabase.table("memory_entries").select("*").execute()
        all_entries = response.data if response.data else []

        results = []
        for entry in all_entries:
            if "embeddings" in entry:
                data_embeddings = json.loads(entry["embeddings"])
                similarity = cosine_similarity(data_embeddings, embeddings[0])
                if similarity > 0.8:  # Adjust similarity threshold as needed
                    results.append({**entry, "similarity": similarity})

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]
    except Exception as e:
        print(f"Error searching memory: {e}")
        return []

# Cosine similarity for comparing embeddings
def cosine_similarity(vec1, vec2):
    import numpy as np
    try:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

# API route for querying
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        user_query = data["query"]

        # Search Supabase for relevant memory
        relevant_memories = search_memory(user_query)
        response = f"Default response to your query: {user_query}"

        log_interaction(user_query, response)
        return jsonify({
            "response": response,
            "relevant_memories": relevant_memories
        })
    except Exception as e:
        print(f"Error in /query route: {e}")
        return jsonify({"error": str(e)}), 500

# API route for fetching all memory (for debugging)
@app.route("/memory", methods=["GET"])
def memory():
    try:
        response = supabase.table("memory_entries").select("*").execute()
        if response.status_code != 200:
            return jsonify({"error": response.json()}), 500
        return jsonify(response.data if response.data else [])
    except Exception as e:
        print(f"Error in /memory route: {e}")
        return jsonify({"error": str(e)}), 500

# Root route for a friendly welcome message
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to GPT Memory Project with Supabase!"}), 200

if __name__ == "__main__":
    # Dynamically bind to the port Render provides or fallback to 22531 for local testing
    port = int(os.environ.get("PORT", 22531))
    app.run(host="0.0.0.0", port=port, debug=True)
