# test_setup.py
import spacy
from sentence_transformers import SentenceTransformer

# Load Spacy model
nlp = spacy.load('en_core_web_sm')
doc = nlp("This is a test sentence for Spacy.")
print("Spacy works:", [token.text for token in doc])

# Load SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["This is a test sentence for SentenceTransformers."])
print("SentenceTransformers works:", embeddings)

