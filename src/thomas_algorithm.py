import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system of equations using the Thomas Algorithm.

    Parameters:
    a (numpy array): Subdiagonal elements (length n-1)
    b (numpy array): Main diagonal elements (length n)
    c (numpy array): Superdiagonal elements (length n-1)
    d (numpy array): Right-hand side vector (length n)

    Returns:
    x (numpy array): Solution vector (length n)
    """
    n = len(b)
    
    # Step 1: LU Decomposition (in-place)
    for i in range(1, n):
        a[i-1] = a[i-1] / b[i-1]  # t_i = a_i / b_{i-1}
        b[i] = b[i] - a[i-1] * c[i-1]  # b_i = b_i - t_i * c_{i-1}
    
    # Step 2: Forward Solve (Ly = d)
    for i in range(1, n):
        d[i] = d[i] - a[i-1] * d[i-1]  # y_i = d_i - t_i * y_{i-1}
    
    # Step 3: Back Solve (Ux = y)
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]  # x_n = y_n / b_n
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]  # x_i = (y_i - c_i * x_{i+1}) / b_i
    
    return x

# Example usage:
n = 5
a = np.array([1, 1, 1, 1])  # Subdiagonal (length n-1)
b = np.array([4, 4, 4, 4, 4])  # Main diagonal (length n)
c = np.array([1, 1, 1, 1])  # Superdiagonal (length n-1)
d = np.array([1, 2, 3, 4, 5])  # Right-hand side vector (length n)

x = thomas_algorithm(a, b, c, d)
print("Solution vector x:", x)