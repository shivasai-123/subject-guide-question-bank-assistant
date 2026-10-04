from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_a = (
    "Minimum support is the minimum frequency threshold "
    "required for an itemset to be considered frequent."
)

sentence_b = (
    "How frequently must an itemset occur to be considered frequent?"
)

sentence_c = (
    "Python is a programming language used to build software."
)

embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)
embedding_c = model.encode(sentence_c)

similarity_ab = cosine_similarity(
    [embedding_a],
    [embedding_b]
)[0][0]

similarity_ac = cosine_similarity(
    [embedding_a],
    [embedding_c]
)[0][0]

print("Similarity between A and B:", similarity_ab)
print("Similarity between A and C:", similarity_ac)