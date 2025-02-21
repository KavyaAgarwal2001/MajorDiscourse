import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft

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

def tdma(a, b, c, d):
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

# Domain and grid
L = 2 * np.pi  # Length of the domain in x-direction
h = 1.0       # Height of the domain
Nx = 64       # Number of grid points in x-direction
Nz = 64       # Number of grid points in z-direction

# Grid spacing
dx = L / Nx
dz = h / Nz

# Grid points
x = np.linspace(0, L, Nx, endpoint=False)
z = np.linspace(0, h, Nz, endpoint=True)

# Create a 2D grid
X, Z = np.meshgrid(x, z, indexing='ij')


# Forcing term
k = 2 * np.pi / L  # Wavenumber
f = np.cos(k * X)  # Forcing term

# Fourier transform of the forcing term
f_hat = fft(f, axis=0)


# Initialize the solution in Fourier space
phi_hat = np.zeros_like(f_hat, dtype=complex)

# Boundary condition at z = h
phi_0 = 1.0

# Wavenumbers in x-direction
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

# Loop over each Fourier mode
for i, kx_val in enumerate(kx):
    if kx_val == 0:
        # Handle the zero mode separately (mean value)
        phi_hat[i, :] = 0  # Adjust based on boundary conditions
    else:
        # Coefficients for the tridiagonal matrix
        a = np.ones(Nz-1) / dz**2  # Subdiagonal (a_2 to a_n)
        b = (-2 / dz**2 - kx_val**2) * np.ones(Nz)  # Main diagonal
        c = np.ones(Nz-1) / dz**2  # Superdiagonal (c_1 to c_{n-1})
        
        # Right-hand side
        rhs = f_hat[i, :].copy()
        rhs[0] = 0  # phi(z=0) = 0
        rhs[-1] = phi_0  # phi(z=h) = phi_0
        
        # Solve the tridiagonal system using TDMA
        phi_hat[i, :] = tdma(a, b, c, rhs)

# Inverse Fourier transform to get the solution in real space
phi = ifft(phi_hat, axis=0).real

# Meshgrid for plotting
X, Z = np.meshgrid(x, z, indexing='ij')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, phi, cmap='viridis', edgecolor='none')
fig.colorbar(surf)
ax.set_title('Solution of Poisson Equation using TDMA')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('Ψ')
plt.show()














