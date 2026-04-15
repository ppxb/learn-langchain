from math import sqrt

vec1 = [0.5, 0.5]
vec2 = [0.3, 0.7]


def get_cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm = sqrt(sum(a**2 for a in vec1) * sum(b**2 for b in vec2))
    return dot / norm if norm != 0 else 0.0


print(get_cosine_similarity(vec1, vec2))
