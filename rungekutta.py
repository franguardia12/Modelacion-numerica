import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, y0, t0, tf, paso, orden):
    pasos = int((tf - t0) / paso)
    t = np.linspace(t0, tf, pasos + 1)
    u = np.zeros(pasos + 1)
    q = np.zeros(orden + 1)
    u[0] = y0

    for i in range(pasos):
        div = 0
        sum = 0
        for j in range(orden):
            if(j == 0 or j == orden - 1):
                q[j] = paso * f(t[i], u[i])
            elif (j == orden - 1):
                q[j] = paso * f(t[i] + paso, u[i] + q[j - 1])
            else:
                q[j] = paso * f(t[i] + paso/2, u[i] + q[j - 1]/2)
            sum += q[j]
        u[i+1] = u[i] + (sum/((orden - 1)*2))
        
    return t, u

def funcion(t, y):
    return -2 * y + 1  # Ejemplo

y0, t0, tf, paso = (0, 0, 5, 0.2)

t, u = runge_kutta(funcion, y0, t0, tf, paso, 2)

# Gráfica de la solución
plt.scatter(t, u, label='Runge Kutta')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Método de Runge Kutta')
plt.grid(True)
plt.show()