import numpy as np
import matplotlib.pyplot as plt

def g_squared(N0, hi, hj, xi, B0):
    numerator = xi * hi**2 * hj**2 * N0**2 + xi * N0 * B0 * (hi**2 + hj**2) + B0**2
    denominator = xi**2 * hi**2 * hj**2 * N0**2 + xi * N0 * B0 * (hi**2 + hj**2) + B0**2
    return numerator / denominator

def plot_g_squared_for_xi(xi_values, N0_values, hi, hj, B0):
    for xi in xi_values:
        g_squared_values = [g_squared(N0, hi, hj, xi, B0) for N0 in N0_values]
        plt.plot(N0_values, g_squared_values, label=r'$\xi={}$'.format(xi))

    plt.xlabel(r'$N_0$')
    plt.ylabel(r'$g^2_{ij}(0)$')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_g_squared_for_xi_fixed_N0(xi_values, N0, hi, hj, B0):
    g_squared_values = [g_squared(N0, hi, hj, xi, B0) for xi in xi_values]
    plt.plot(xi_values, g_squared_values, label=r'$N_0={}$'.format(N0))

    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$g^2_{ij}(0)$')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
xi_values = np.linspace(0, 1, 100)
B0 = 1.0
hi = np.sqrt(0.1)
hj = np.sqrt(0.1)
N0_fixed = 5.0
plot_g_squared_for_xi_fixed_N0(xi_values, N0_fixed, hi, hj, B0)

xi_values = [0.2,0.5,1.0]
N0_values = np.linspace(0,100,100)
plot_g_squared_for_xi(xi_values, N0_values, hi, hj, B0)

