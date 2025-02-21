import numpy as np

def gauss_elimination(a, b, c, d):
    n = len(d)
    # Forward elimination with in-place modification of a
    for i in range(1, n):
        a[i-1] = a[i-1] / b[i-1]
        b[i] = b[i] - a[i-1] * c[i-1]
        d[i] = d[i] - a[i-1] * d[i-1]

    # Backward substitution
    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

# Example usage:
n = 5
a = np.array([1, 1, 1, 1])  # Subdiagonal (length n-1)
b = np.array([4, 4, 4, 4, 4])  # Main diagonal (length n)
c = np.array([1, 1, 1, 1])  # Superdiagonal (length n-1)
d = np.array([1, 2, 3, 4, 5])  # Right-hand side vector (length n)

x = gauss_elimination(a, b, c, d)
print("Solution vector x:", x)