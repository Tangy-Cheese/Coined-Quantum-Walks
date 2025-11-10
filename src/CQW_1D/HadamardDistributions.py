import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
 
def psi_up(x, t):
    a_1 = (1+ ((-1)**(t+x)))
    a_2 = (2**((t+3)/2) * (2*np.pi))
    a = a_1 / a_2
    def integrand(k):
        return (np.exp(-1j*k*x) * ((1 + np.cos(k)**2)**(1/2) - 1j*np.sin(k))**(t) * (1 + (np.cos(k))/(1+(np.cos(k))**2) + (1j*np.exp(-1j*k) / (1+(np.cos(k))**2)) ))
    real_part = quad(lambda k: np.real(integrand(k)), -np.pi, np.pi)[0]
    imag_part = quad(lambda k: np.imag(integrand(k)), -np.pi, np.pi)[0]
    return a*(real_part + 1j * imag_part)
 
def psi_down(x, t):
    a_1 = (1+ ((-1)**(t+x)))
    a_2 = (2**((t+3)/2) * (2*np.pi))
    a = a_1 / a_2
    def integrand(k):
        omega = np.cos(k)
        return (np.exp(-1j*k*x) * ((1 + np.cos(k)**2)**(1/2) - 1j*np.sin(k))**(t) * (1j*(1 - (np.cos(k))/(1+(np.cos(k))**2)) + (np .exp(1j*k) / (1+(np.cos(k))**2))) )
    real_part = quad(lambda k: np.real(integrand(k)), -np.pi, np.pi)[0]
    imag_part = quad(lambda k: np.imag(integrand(k)), -np.pi, np.pi)[0]
    return a*(real_part + 1j * imag_part)
 
def probability_distribution(x, t):
    up = psi_up(x, t)
    down = psi_down(x, t)
    return np.abs(up)**2 + np.abs(down)**2
 
x_values = np.linspace(-5,5,11)
time_values = [4]
 
# Plot the probability distribution for different time steps
for t in time_values:
    y_values = [probability_distribution(x, t) for x in x_values]
    plt.plot(x_values, y_values, label=f't = {t}')
 
plt.title('Probability distribution of the Hadamard quantum walk')
plt.xlabel('Position (x)')
plt.ylabel('Probability')
plt.legend()
plt.ylim(0, 1)
plt.show()