import numpy as np
import matplotlib.pyplot as plt

# Variables del modelo
g = 3.72
M = 1000
H0 = 130_000
v0 = -5555
ALFA = 150_000
BETA = 0.0192
Hap = 13_000
ETA = 0.3600
Hlib = 2000
ve = 1000
Hgrua = 20

k1 = 50
k2 = 50
mc0 = M*g/ve  #caudal al final de la etapa 3

#función que aplica el método de Runge-Kutta de orden n
def runge_kutta(f, y0, t0, tf, paso, orden):
    pasos = int((tf - t0) / paso)
    t = np.linspace(t0, tf, pasos + 1)
    u = np.zeros((pasos + 1, len(y0)))
    k = np.zeros((orden + 1, len(y0)))
    u[0] = y0

    for i in range(pasos):
        sum = 0
        for j in range(orden):
            if(j == 0):
                k[j] = f(t[i], u[i])
            elif (j == orden - 1):
                k[j] = f(t[i] + paso, u[i] + k[j - 1])
            else:
                k[j] = 2 * f(t[i] + paso/2, u[i] + k[j - 1]/2)
            sum += k[j]
        u[i+1] = u[i] + paso * (sum/((orden - 1)*2))

    return t, u

# Método de trapecios compuesto
def trapecios_compuesto(y, h):
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

#función que aplica la ecuación diferencial de la etapa 1
def etapa1(t, yv):
    y, v = yv
    dy_dt = v
    dv_dt = -g + (BETA / M) * np.exp(-y / ALFA) * v**2
    return np.array([dy_dt, dv_dt])

#función que aplica la ecuación diferencial de la etapa 2
def etapa2(t, yv):
    y, v = yv
    dy_dt = v
    dv_dt = -g + (ETA/M) * v**2
    return np.array([dy_dt, dv_dt])

#función que aplica la ecuación diferencial de la etapa 3
def etapa3(t, yvm):
    y, v, m = yvm
    dy_dt = v
    dm_dt = mc0 - k1 * (y - Hgrua) - k2 * v
    if dm_dt < 0:
        dm_dt = 0
    dv_dt = -g + (1 / M)  * dm_dt * (ve - v)
    return np.array([dy_dt, dv_dt, dm_dt])

# Condiciones iniciales
y0 = [H0, v0]
t01, tf1= 0, 42.19
paso = 0.01
orden = 2

# Etapa 1
t1, yv1 = runge_kutta(etapa1, y0, t01, tf1, paso, orden)
y1 = yv1[:, 0]
v1 = yv1[:, 1]
#velocidadFinalEtapaI = v1[-1]

# Etapa 2
t02 = tf1
tf2 = 92.01
t2, yv2 = runge_kutta(etapa2, yv1[-1], t02, tf2, paso, orden)
y2 = yv2[:, 0]
v2 = yv2[:, 1]
#velocidadFinalEtapaII = v2[-1]

# Etapa 3
t03 = tf2
tf3 = 114
mc3 = mc0 - k1*(y2[-1] - Hgrua) - k2*v2[-1]
if mc3 < 0:
    mc3 = 0
y03 = (yv2[-1][0], yv2[-1][1], mc3)
t3, yvm3 = runge_kutta(etapa3, y03, t03, tf3, paso, orden)
y3 = yvm3[:, 0]
v3 = yvm3[:, 1]
m3 = yvm3[:, 2]
#Tiempo abs(v3) <= 0.03 = 113.28 seg; k1 = 50, k2 = 50
#velocidadFinalEtapaIII = v3[-1]

# Masa total en kg utilizada
m_total = round(trapecios_compuesto(m3, paso), 2)
#print(f"Masa total en kg: {m_total}")

#---------------------------------GRÁFICOS DE ETAPAS 1, 2 Y 3---------------------------------


# Concatenar resultados
t_total = np.concatenate((t1, t2[1:], t3[1:]))
y_total = np.concatenate((y1, y2[1:], y3[1:]))
v_total = np.concatenate((v1, v2[1:], v3[1:]))

# Gráfica
plt.figure(figsize=(12, 6))

# Posición
plt.subplot(2, 1, 1)
plt.plot(t_total, y_total, label='Posición (y)', color= 'black')
plt.ylabel('Posición (m)')
plt.axvline(x=tf1, color='y', linestyle='--', label='Inicio etapa II')
plt.axvline(x=tf2, color='g', linestyle='--', label='Inicio etapa III')
plt.legend()
plt.grid(True)

# Velocidad
plt.subplot(2, 1, 2)
plt.plot(t_total, v_total, label='Velocidad (v)', color='blue')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.axvline(x=tf1, color='y', linestyle='--', label='Inicio etapa II')
plt.axvline(x=tf2, color='g', linestyle='--', label='Inicio etapa III')
plt.legend()
plt.grid(True)
plt.suptitle('Modelo de descenso del Rover Perseverance en Marte')
plt.show()

#---------------------------------GRÁFICOS DE ETAPA 3-----------------------------------------


t_etapa3 = t3[1:]
y_etapa3 = y3[1:]
v_etapa3 = v3[1:]

# Gráfica
plt.figure(figsize=(12, 9))

# Posición
plt.subplot(3, 1, 1)
plt.plot(t3[1:], y3[1:], label='Posición (y)', color= 'black')
plt.ylabel('Posición (m)')
plt.axvline(x=tf2, color='g', linestyle='--', label='Inicio etapa III')
plt.legend()
plt.grid(True)

# Velocidad
plt.subplot(3, 1, 2)
plt.plot(t3[1:], v3[1:], label='Velocidad (v)', color='blue')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.axvline(x=tf2, color='g', linestyle='--', label='Inicio etapa III')
plt.legend()
plt.grid(True)

# Caudal másico
plt.subplot(3, 1, 3)
plt.plot(t3[1:], m3[1:], label='Caudal másico (m)', color='red')
plt.xlabel('Tiempo (s)')
plt.ylabel('Caudal másico (kg/s)')
plt.axvline(x=tf2, color='g', linestyle='--', label='Inicio etapa III')
plt.legend()
plt.grid(True)
plt.suptitle('Modelo de descenso del Rover Perseverance en Marte (Etapa III)')
plt.show()
