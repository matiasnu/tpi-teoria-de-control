import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ==========================================
# CONFIGURACIÓN DE LA SIMULACIÓN (Teoría)
# ==========================================
# Modelo del Sistema:
# Controlador: PD (Proporcional + Derivativo)
# Actuador (Token Bucket/Escalador): Aporta efecto integrador (acumula cambios).
# Planta (Microservicio): Sistema de primer orden (retardo en procesamiento).

class SimuladorRateLimiter:
    def __init__(self):
        self.dt = 0.05  # Paso de tiempo
        self.sim_time = 30.0 # Duración
        self.t = np.arange(0, self.sim_time, self.dt)
        self.n = len(self.t)
        
        # Valores iniciales del controlador
        self.Kp = 0.5
        self.Kd = 0.2
        
        # Escenario actual
        self.escenario = 'Rafagas' # 'Rafagas' o 'DoS'

    def generar_escenario(self):
        """Genera las señales de Referencia (Setpoint) y Perturbación (D)"""
        # Referencia (R): Nivel de servicio deseado (ej. 100 req/s)
        R = np.ones(self.n) * 100.0 
        
        # Perturbación (D): Tráfico entrante NO deseado o carga extra
        D = np.zeros(self.n)
        
        if self.escenario == 'Rafagas':
            # Ráfagas cortas en t=5s y t=15s
            D[int(5/self.dt):int(7/self.dt)] = 150.0  # Pico fuerte
            D[int(15/self.dt):int(18/self.dt)] = 80.0 # Pico medio
            
        elif self.escenario == 'DoS':
            # Ataque sostenido desde t=5s hasta el final
            D[int(5/self.dt):] = 400.0 # Carga masiva constante
            
        return R, D

    def ejecutar_simulacion(self):
        R, D = self.generar_escenario()
        
        # Inicialización de vectores
        Y = np.zeros(self.n) # Salida (Throughput real)
        e = np.zeros(self.n) # Error
        u = np.zeros(self.n) # Señal de control
        
        # Variables de estado del sistema
        capacidad_actual = 0.0 # Variable interna del actuador (Efecto memoria/Bucket)
        y_prev = 0.0
        e_prev = 0.0
        
        # Bucle de simulación
        for i in range(1, self.n):
            # 1. Medición y Cálculo del Error
            # H(s) = 1 (Sensor ideal)
            e[i] = R[i] - y_prev
            
            # 2. Controlador PD
            # u(t) = Kp*e(t) + Kd*de(t)/dt
            derivativa = (e[i] - e_prev) / self.dt
            u[i] = (self.Kp * e[i]) + (self.Kd * derivativa)
            
            # 3. Actuador con "Memoria" (Token Bucket / Autoscaler)
            # Tal como se explicó en el TP modificado, el actuador acumula o integra
            # la señal de control, comportándose como un sistema Tipo 1.
            # Nueva capacidad = Capacidad anterior + ajuste del controlador
            capacidad_actual += u[i] * self.dt * 5.0 # Factor de ganancia del actuador
            
            # Limitaciones físicas (No puede haber capacidad negativa)
            if capacidad_actual < 0: capacidad_actual = 0
            
            # 4. Planta (Microservicio) + Perturbación
            # La perturbación D se suma a la capacidad intentando desbordar, 
            # pero el sistema solo procesa lo que la planta permite.
            # Modelamos la planta como un lag de primer orden hacia la capacidad objetivo + perturbación
            # Y(s)/U(s) = 1/(tau*s + 1)
            tau = 0.5 # Constante de tiempo del servidor (lag)
            
            entrada_planta = capacidad_actual - D[i] * 0.2 # La perturbación degrada la capacidad efectiva
            
            # Ecuación en diferencias para lag de primer orden
            Y[i] = (self.dt * entrada_planta + tau * y_prev) / (tau + self.dt)
            
            # Actualizar previos
            y_prev = Y[i]
            e_prev = e[i]
            
        return self.t, R, Y, u, e, D

# ==========================================
# INTERFAZ GRÁFICA (Matplotlib)
# ==========================================

sim = SimuladorRateLimiter()

# Crear figura y ejes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.4)

# Configuración de gráficas iniciales
l_ref, = ax1.plot([], [], 'k--', label=r'$\theta_i$ (Setpoint / R)', linewidth=1.5)
l_y, = ax1.plot([], [], 'b-', label='Respuesta del Sistema (Y)', linewidth=2)
l_pert, = ax1.plot([], [], 'r:', label='Perturbación (D)', alpha=0.6)

l_err, = ax2.plot([], [], 'g-', label='Error (e)', linewidth=1.5)
l_u, = ax3.plot([], [], 'm-', label='Salida Controlador (u)', linewidth=1.5)

# Etiquetas y Títulos
ax1.set_title('Respuesta Temporal del Sistema (Rate Limiter Controlado)')
ax1.set_ylabel('Throughput (req/s)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.set_ylabel('Error $e(t)$')
ax2.grid(True, linestyle=':', alpha=0.6)

ax3.set_ylabel('Señal de Control $u(t)$')
ax3.set_xlabel('Tiempo (s)')
ax3.grid(True, linestyle=':', alpha=0.6)

# Función de actualización
def update(val):
    # Leer sliders
    sim.Kp = s_kp.val
    sim.Kd = s_kd.val
    
    # Recalcular
    t, R, Y, u, e, D = sim.ejecutar_simulacion()
    
    # Actualizar datos en gráficas
    l_ref.set_data(t, R)
    l_y.set_data(t, Y)
    l_pert.set_data(t, D) # Escalada para visualización si es necesario
    l_err.set_data(t, e)
    l_u.set_data(t, u)
    
    # Reajustar escalas dinámicamente
    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    ax3.relim(); ax3.autoscale_view()
    
    fig.canvas.draw_idle()

# Sliders (Controles Dinámicos)
ax_kp = plt.axes([0.15, 0.07, 0.62, 0.03], facecolor='lightgoldenrodyellow')
ax_kd = plt.axes([0.15, 0.03, 0.62, 0.03], facecolor='lightgoldenrodyellow')

s_kp = Slider(ax_kp, 'Kp (Prop.)', 0.0, 5.0, valinit=0.5, valstep=0.1)
s_kd = Slider(ax_kd, 'Kd (Deriv.)', 0.0, 5.0, valinit=0.2, valstep=0.1)

s_kp.on_changed(update)
s_kd.on_changed(update)

# Botones de Escenario
rax = plt.axes([0.82, 0.05, 0.15, 0.15], facecolor='#f0f0f0')
radio = RadioButtons(rax, ('Rafagas', 'DoS'))

def change_scenario(label):
    sim.escenario = label
    update(None)

radio.on_clicked(change_scenario)

# Ejecución inicial
update(None)
plt.show()