import json
from sentence_transformers import SentenceTransformer, util
import spacy

# File to store chat history
HISTORY_FILE = "chat_history.json"

# Load SpaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load chat history
def load_history():
    try:
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Function to save chat history
def save_history(history):
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

# Function to find context based on semantic similarity
def get_context(query, history):
    if not history:
        return None

    query_embedding = model.encode(query, convert_to_tensor=True)
    best_match = None
    highest_similarity = 0.0

    for entry in history:
        past_query_embedding = model.encode(entry['query'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, past_query_embedding).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entry

    return best_match if highest_similarity > 0.5 else None

def main():
    print("Real-Time Memory System initialized. Listening for queries...")
    history = load_history()

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting the system. Goodbye!")
            break

        # Tokenize the input query (example usage of SpaCy)
        tokens = [token.text for token in nlp(user_query)]
        print(f"Tokenized Query: {tokens}")

        # Retrieve context
        context = get_context(user_query, history)
        if context:
            print(f"Found Relevant Context: {context['query']}")
            response = f"Based on your past query '{context['query']}', here's my response: {context['response']}"
        else:
            response = f"Default response to your query: {user_query}"

        # Show response
        print(f"\nResponse:\n{response}")

        # Update history
        history.append({'query': user_query, 'response': response})
        save_history(history)

if __name__ == "__main__":
    main()
