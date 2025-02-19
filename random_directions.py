import  numpy as np

def random_directions(n):
    """Generate random unit vectors in Cartesian coordinates"""
    vectors = np.random.normal(size=(n, 3))  # Sample x, y and z from normal distribution
    vectors /= np.linalg.norm(vecs, axis=1, keepdims=True)  # Normalise to magnitude=1
    return vectors

# Example usage:
directions = random_directions(5)
print(directions)
