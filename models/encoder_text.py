from sentence_transformers import SentenceTransformer

sentences = ["Hi, I am Arjun"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode(sentences)
print(embedding.shape)