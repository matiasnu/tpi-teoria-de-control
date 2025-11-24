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
        
        # Parámetros de escalado de instancias
        self.escalado_habilitado = True
        self.capacidad_por_instancia = 50.0  # req/s por instancia
        self.instancias_min = 1
        self.instancias_max = 10
        self.delay_escalado = 3.0  # Segundos de retardo para aplicar cambio
        self.overhead_coordinacion = 0.05  # Factor de overhead por instancia adicional

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
        instancias = np.zeros(self.n) # Número de instancias activas
        capacidad_total = np.zeros(self.n) # Capacidad total disponible
        costo_acumulado = np.zeros(self.n) # Costo operativo
        
        # Variables de estado del sistema
        capacidad_actual = 0.0 # Variable interna del actuador (Efecto memoria/Bucket)
        y_prev = 0.0
        e_prev = 0.0
        
        # Variables de escalado
        instancias_running = self.instancias_min
        instancias_target = self.instancias_min
        tiempo_desde_cambio = 0.0
        costo_total = 0.0
        
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
            
            # 3.1. Lógica de Escalado de Instancias (si está habilitado)
            if self.escalado_habilitado:
                # Calcular instancias necesarias basado en capacidad requerida
                instancias_requeridas = capacidad_actual / self.capacidad_por_instancia
                
                # Cuantización: las instancias son discretas
                instancias_target = int(np.ceil(instancias_requeridas))
                
                # Aplicar límites
                instancias_target = np.clip(instancias_target, self.instancias_min, self.instancias_max)
                
                # Aplicar retardo de escalado (simula tiempo de arranque/apagado)
                tiempo_desde_cambio += self.dt
                if instancias_target != instancias_running and tiempo_desde_cambio >= self.delay_escalado:
                    instancias_running = instancias_target
                    tiempo_desde_cambio = 0.0
                
                # Capacidad efectiva basada en instancias running
                capacidad_efectiva = instancias_running * self.capacidad_por_instancia
            else:
                # Sin escalado: usar capacidad directa del actuador
                capacidad_efectiva = capacidad_actual
                instancias_running = max(1, int(capacidad_actual / self.capacidad_por_instancia))
            
            # Registrar para gráficos
            instancias[i] = instancias_running
            capacidad_total[i] = capacidad_efectiva
            
            # 3.2. Calcular costo operativo
            costo_total += instancias_running * self.dt * 0.1  # $0.1 por instancia por segundo
            costo_acumulado[i] = costo_total
            
            # 4. Planta (Microservicio) + Perturbación
            # La perturbación D se suma a la capacidad intentando desbordar, 
            # pero el sistema solo procesa lo que la planta permite.
            # Modelamos la planta como un lag de primer orden hacia la capacidad objetivo + perturbación
            # Y(s)/U(s) = 1/(tau*s + 1)
            tau = 0.5 # Constante de tiempo del servidor (lag)
            
            # Overhead de coordinación: más instancias = más overhead
            tau_efectivo = tau * (1 + self.overhead_coordinacion * (instancias_running - 1))
            
            entrada_planta = capacidad_efectiva - D[i] * 0.2 # La perturbación degrada la capacidad efectiva
            
            # Ecuación en diferencias para lag de primer orden
            Y[i] = (self.dt * entrada_planta + tau_efectivo * y_prev) / (tau_efectivo + self.dt)
            
            # Actualizar previos
            y_prev = Y[i]
            e_prev = e[i]
            
        return self.t, R, Y, u, e, D, instancias, capacidad_total, costo_acumulado

# ==========================================
# INTERFAZ GRÁFICA (Matplotlib)
# ==========================================

sim = SimuladorRateLimiter()

# Crear figura y ejes (4 subplots para incluir instancias)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.4)

# Configuración de gráficas iniciales
l_ref, = ax1.plot([], [], 'k--', label=r'$\theta_i$ (Setpoint / R)', linewidth=1.5)
l_y, = ax1.plot([], [], 'b-', label='Respuesta del Sistema (Y)', linewidth=2)
l_pert, = ax1.plot([], [], 'r:', label='Perturbación (D)', alpha=0.6)

l_err, = ax2.plot([], [], 'g-', label='Error (e)', linewidth=1.5)
l_u, = ax3.plot([], [], 'm-', label='Salida Controlador (u)', linewidth=1.5)

# Nuevo subplot para instancias (usa estilo escalonado para mejor visualización)
l_inst, = ax4.plot([], [], 'c-', label='Instancias Activas', linewidth=3, marker='s', 
                    markersize=6, markerfacecolor='cyan', markeredgecolor='darkblue', 
                    markeredgewidth=1.5, drawstyle='steps-post')

# Crear eje Y secundario para capacidad
ax4_cap = ax4.twinx()
l_cap, = ax4_cap.plot([], [], 'orange', label='Capacidad Total (req/s)', linewidth=2.5, 
                      linestyle='--', alpha=0.8)

# Etiquetas y Títulos
ax1.set_title('Respuesta Temporal del Sistema (Rate Limiter + Autoscaling)')
ax1.set_ylabel('Throughput (req/s)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.set_ylabel('Error $e(t)$')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

ax3.set_ylabel('Señal de Control $u(t)$')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')

ax4.set_ylabel('Instancias (#)', color='cyan', fontweight='bold')
ax4.tick_params(axis='y', labelcolor='cyan')
ax4.set_xlabel('Tiempo (s)')
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.set_ylim([0, 11])  # Rango fijo para instancias (0-11)

ax4_cap.set_ylabel('Capacidad Total (req/s)', color='orange', fontweight='bold')
ax4_cap.tick_params(axis='y', labelcolor='orange')

# Leyendas combinadas
lines_4 = [l_inst, l_cap]
labels_4 = ['Instancias Activas', 'Capacidad Total']
ax4.legend(lines_4, labels_4, loc='upper left')

# Función de actualización
def update(val):
    # Leer sliders
    sim.Kp = s_kp.val
    sim.Kd = s_kd.val
    
    # Recalcular
    t, R, Y, u, e, D, instancias, capacidad_total, costo_acumulado = sim.ejecutar_simulacion()
    
    # Actualizar datos en gráficas
    l_ref.set_data(t, R)
    l_y.set_data(t, Y)
    l_pert.set_data(t, D)
    l_err.set_data(t, e)
    l_u.set_data(t, u)
    l_inst.set_data(t, instancias)
    l_cap.set_data(t, capacidad_total)
    
    # Reajustar escalas dinámicamente
    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    ax3.relim(); ax3.autoscale_view()
    
    # Para ax4: mantener escala fija de instancias (0-11) pero ajustar capacidad
    ax4_cap.relim()
    ax4_cap.autoscale_view()
    
    # Agregar texto con costo final e instancias actuales
    max_inst = int(np.max(instancias))
    ax4.set_title(f'Escalado de Instancias (Max: {max_inst}) | Costo Total: ${costo_acumulado[-1]:.2f}', 
                  fontsize=10, fontweight='bold')
    
    fig.canvas.draw_idle()

# Sliders (Controles Dinámicos)
ax_kp = plt.axes([0.15, 0.13, 0.62, 0.02], facecolor='lightgoldenrodyellow')
ax_kd = plt.axes([0.15, 0.10, 0.62, 0.02], facecolor='lightgoldenrodyellow')
ax_cap = plt.axes([0.15, 0.07, 0.62, 0.02], facecolor='lightblue')
ax_delay = plt.axes([0.15, 0.04, 0.62, 0.02], facecolor='lightblue')

s_kp = Slider(ax_kp, 'Kp (Prop.)', 0.0, 5.0, valinit=0.5, valstep=0.1)
s_kd = Slider(ax_kd, 'Kd (Deriv.)', 0.0, 5.0, valinit=0.2, valstep=0.1)
s_cap = Slider(ax_cap, 'Cap/Inst', 30.0, 150.0, valinit=50.0, valstep=10.0)
s_delay = Slider(ax_delay, 'Delay Esc.', 0.5, 10.0, valinit=3.0, valstep=0.5)

def update_params(val):
    sim.capacidad_por_instancia = s_cap.val
    sim.delay_escalado = s_delay.val
    update(None)

s_kp.on_changed(update)
s_kd.on_changed(update)
s_cap.on_changed(update_params)
s_delay.on_changed(update_params)

# Botones de Escenario y Escalado
rax = plt.axes([0.82, 0.08, 0.15, 0.12], facecolor='#f0f0f0')
radio = RadioButtons(rax, ('Rafagas', 'DoS'))

rax2 = plt.axes([0.82, 0.01, 0.15, 0.06], facecolor='#e0f0ff')
radio_escalado = RadioButtons(rax2, ('Escalado ON', 'Escalado OFF'))

def change_scenario(label):
    sim.escenario = label
    update(None)

def toggle_escalado(label):
    sim.escalado_habilitado = (label == 'Escalado ON')
    update(None)

radio.on_clicked(change_scenario)
radio_escalado.on_clicked(toggle_escalado)

# Ejecución inicial
update(None)
plt.show()