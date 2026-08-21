from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Minimum support is the minimum frequency threshold required for an itemset to be considered frequent."

embedding = model.encode(text)

print("Embedding created!")
print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])
