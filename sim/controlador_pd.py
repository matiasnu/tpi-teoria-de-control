import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

# ==========================================
# CONFIGURACIÓN DE LA SIMULACIÓN (Teoría)
# ==========================================
# Modelo del Sistema:
# Controlador: PD (Proporcional + Derivativo)
# Actuador (Token Bucket/Escalador): Aporta efecto integrador (acumula cambios).
# Planta (Microservicio): Sistema de primer orden (retardo en procesamiento).

class SimuladorRateLimiter:
    def __init__(self):
        self.dt = 0.05
        self.sim_time = 30.0
        self.t = np.arange(0, self.sim_time, self.dt)
        self.n = len(self.t)

        # Controlador
        self.Kp = 0.5
        self.Kd = 0.2

        # Setpoint
        self.setpoint = 100.0

        # Perturbaciones
        self.tiempo_inicio_pert1 = 5.0
        self.duracion_pert1 = 3.0
        self.intensidad_pert1 = 150.0

        self.tiempo_inicio_pert2 = 15.0
        self.duracion_pert2 = 3.0
        self.intensidad_pert2 = 100.0

        # Escenario
        self.escenario = 'Rafagas'

        # --- NUEVOS PARÁMETROS ---
        # Actuador / límites físicos
        self.capacidad_max = 500.0     # tope físico (tokens/req/s equivalentes)
        self.capacidad_min = 0.0

        # Anti-windup: 'freeze' or 'backcalc' (string)
        self.anti_windup = 'freeze'

        # Derivative filter time constant (segundos)
        self.tau_d = 0.1

        # Ruido y retardo en la medición
        self.measurement_noise_std = 0.0
        self.measurement_delay = 0.0  # en s (si >0 se implementa en muestras)

        # Cuantización de capacidad (por ejemplo, 1 = 1 instancia/token discreto)
        self.quantization_step = 1.0

        # Actuator gain (como antes)
        self.actuator_gain = 5.0

    def generar_escenario(self):
        R = np.ones(self.n) * self.setpoint
        D = np.zeros(self.n)
        if self.escenario == 'Rafagas':
            idx_inicio1 = int(self.tiempo_inicio_pert1 / self.dt)
            idx_fin1 = int((self.tiempo_inicio_pert1 + self.duracion_pert1) / self.dt)
            idx_inicio1 = max(0, min(idx_inicio1, self.n - 1))
            idx_fin1 = max(idx_inicio1, min(idx_fin1, self.n))
            D[idx_inicio1:idx_fin1] = self.intensidad_pert1

            idx_inicio2 = int(self.tiempo_inicio_pert2 / self.dt)
            idx_fin2 = int((self.tiempo_inicio_pert2 + self.duracion_pert2) / self.dt)
            idx_inicio2 = max(0, min(idx_inicio2, self.n - 1))
            idx_fin2 = max(idx_inicio2, min(idx_fin2, self.n))
            D[idx_inicio2:idx_fin2] = self.intensidad_pert2

        elif self.escenario == 'DoS':
            idx_inicio = int(self.tiempo_inicio_pert1 / self.dt)
            idx_fin = int((self.tiempo_inicio_pert1 + self.duracion_pert1) / self.dt)
            idx_inicio = max(0, min(idx_inicio, self.n - 1))
            idx_fin = max(idx_inicio, min(idx_fin, self.n))
            D[idx_inicio:idx_fin] = self.intensidad_pert1

        return R, D

    def ejecutar_simulacion(self, return_metrics=True):
        R, D = self.generar_escenario()

        Y = np.zeros(self.n)
        e = np.zeros(self.n)
        u = np.zeros(self.n)
        capacidad = np.zeros(self.n)  # registro temporal de capacidad (actuador)
        measured_Y = np.zeros(self.n)

        # variables internas
        capacidad_actual = 0.0
        y_prev = 0.0
        e_prev = 0.0

        # derivative filter state
        d_filtered = 0.0
        d_prev = 0.0

        # measurement delay buffer (en samples)
        delay_samples = int(np.round(self.measurement_delay / self.dt)) if self.measurement_delay > 0 else 0
        meas_buffer = [0.0] * (delay_samples + 1)

        # parámetros planta
        tau = 0.5

        for i in range(1, self.n):
            # medición con delay + ruido (simulamos sensor ideal pero con delay/noise)
            # la medición que utiliza el controlador es la salida medida afectada por delay/noise
            meas_buffer.pop(0)
            meas_buffer.append(y_prev + np.random.randn() * self.measurement_noise_std)
            y_measured = meas_buffer[0]

            # error y control (sensor ideal vs medido: en TP asumías ideal, aquí mostramos medido)
            e[i] = R[i] - y_measured

            # derivada filtrada (forma discreta del filtro de primer orden sobre la derivada)
            derivative_raw = (e[i] - e_prev) / self.dt
            # filtro exponencial simple (equivale a RLP con tau_d)
            alpha = self.tau_d / (self.tau_d + self.dt) if self.tau_d > 0 else 0.0
            d_filtered = alpha * d_prev + (1 - alpha) * derivative_raw

            u[i] = (self.Kp * e[i]) + (self.Kd * d_filtered)

            # ACTUADOR: integración con ganancia y anti-windup + cuantización + saturación
            delta_cap = u[i] * self.dt * self.actuator_gain

            # anti-windup policy
            prospective_cap = capacidad_actual + delta_cap
            # saturación aplicada
            prospective_cap = max(self.capacidad_min, min(self.capacidad_max, prospective_cap))
            if self.anti_windup == 'freeze':
                # Si el actuador está saturado y la orden llevaría fuera del rango, no integrar
                if (capacidad_actual <= self.capacidad_min and delta_cap < 0) or \
                   (capacidad_actual >= self.capacidad_max and delta_cap > 0):
                    # No integrar (acumulador queda igual)
                    pass
                else:
                    capacidad_actual = prospective_cap
            elif self.anti_windup == 'backcalc':
                # Simple back-calculation: permito integración pero la reduzco cuando saturado
                if prospective_cap != capacidad_actual + delta_cap:  # hay saturación
                    # reducir la contribución (factor de corrección)
                    back_gain = 0.5
                    capacidad_actual += back_gain * delta_cap
                    capacidad_actual = max(self.capacidad_min, min(self.capacidad_max, capacidad_actual))
                else:
                    capacidad_actual = prospective_cap
            else:
                capacidad_actual = prospective_cap

            # cuantización (simula instancias discretas / tokens)
            if self.quantization_step and self.quantization_step > 0:
                capacidad_actual = np.round(capacidad_actual / self.quantization_step) * self.quantization_step

            capacidad[i] = capacidad_actual

            # entrada efectiva a la planta (capacidad - parte de perturbación que degrada)
            entrada_planta = capacidad_actual - D[i] * 0.2

            # lag primer orden (ecuación en diferencias)
            Y[i] = (self.dt * entrada_planta + tau * y_prev) / (tau + self.dt)

            # almacenar medición
            measured_Y[i] = y_measured

            # actualizar previos
            y_prev = Y[i]
            e_prev = e[i]
            d_prev = d_filtered

        # métricas básicas
        metrics = {}
        if return_metrics:
            # error en estado estacionario: promedio de últimos 10% de la simulación
            tail = int(self.n * 0.1)
            ess = np.mean(R[-tail:] - Y[-tail:])
            overshoot = (np.max(Y) - np.max(R)) / np.max(R) * 100.0 if np.max(R) > 0 else 0.0
            max_u = np.max(np.abs(u))
            # settling time: primer t donde señal dentro del 2% del setpoint y se mantiene
            tol = 0.02 * R[0]
            settling_time = None
            steady_band = np.abs(R - Y) <= tol
            for idx in range(int(self.n*0.1), self.n):
                if np.all(steady_band[idx:]):
                    settling_time = self.t[idx]
                    break
            metrics = {
                'ess': float(ess),
                'overshoot_pct': float(overshoot),
                'max_control_effort': float(max_u),
                'settling_time_s': float(settling_time) if settling_time is not None else None
            }

        return self.t, R, Y, u, e, D, capacidad, measured_Y, metrics


# ==========================================
# INTERFAZ GRÁFICA (Matplotlib)
# ==========================================

sim = SimuladorRateLimiter()

# Crear figura y ejes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.34, right=0.95, hspace=0.25)

# Configuración de gráficas iniciales
l_ref, = ax1.plot([], [], 'k--', label=r'$\theta_i$ (Setpoint / R)', linewidth=1.5)
l_y, = ax1.plot([], [], 'b-', label='Respuesta del Sistema (Y)', linewidth=2)
l_pert, = ax1.plot([], [], 'r:', label='Perturbación (D)', alpha=0.6)

l_err, = ax2.plot([], [], 'g-', label='Error (e)', linewidth=1.5)
l_u, = ax3.plot([], [], 'm-', label='Salida Controlador (u)', linewidth=1.5)

# Etiquetas y Títulos
ax1.set_title('Respuesta Temporal del Sistema (Rate Limiter Controlado)')
ax1.set_ylabel('Throughput (req/s)')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.set_ylabel('Error $e(t)$')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.6)

ax3.set_ylabel('Control $u(t)$')
ax3.set_xlabel('Tiempo (s)')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, linestyle=':', alpha=0.6)

# Función de actualización
def update(val):
    # Leer sliders del controlador
    sim.Kp = s_kp.val
    sim.Kd = s_kd.val
    
    # Leer sliders de perturbación - Ráfaga 1
    sim.tiempo_inicio_pert1 = s_tiempo_pert1.val
    sim.duracion_pert1 = s_duracion_pert1.val
    sim.intensidad_pert1 = s_intensidad_pert1.val
    
    # Leer sliders de perturbación - Ráfaga 2
    sim.tiempo_inicio_pert2 = s_tiempo_pert2.val
    sim.duracion_pert2 = s_duracion_pert2.val
    sim.intensidad_pert2 = s_intensidad_pert2.val
    
    # Recalcular (sin calcular métricas para mayor eficiencia)
    t, R, Y, u, e, D, capacidad, measured_Y, _ = sim.ejecutar_simulacion(return_metrics=False)
    
    # Actualizar datos en gráficas
    l_ref.set_data(t, R)
    l_y.set_data(t, Y)
    l_pert.set_data(t, D)
    l_err.set_data(t, e)
    l_u.set_data(t, u)
    
    # Reajustar escalas dinámicamente
    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    ax3.relim(); ax3.autoscale_view()
    
    fig.canvas.draw_idle()

# Sliders (Controles Dinámicos)
# Sliders del controlador (arriba)
ax_kp = plt.axes([0.15, 0.29, 0.62, 0.02], facecolor='lightgoldenrodyellow')
ax_kd = plt.axes([0.15, 0.26, 0.62, 0.02], facecolor='lightgoldenrodyellow')

s_kp = Slider(ax_kp, 'Kp (Prop.)', 0.0, 5.0, valinit=0.5, valstep=0.1)
s_kd = Slider(ax_kd, 'Kd (Deriv.)', 0.0, 5.0, valinit=0.2, valstep=0.1)

# TextBox de Setpoint (entrada directa)
ax_setpoint = plt.axes([0.15, 0.235, 0.15, 0.02], facecolor='lightcyan')
tbox_setpoint = TextBox(ax_setpoint, 'Setpoint:', initial='100.0')

# Sliders de Perturbación - Ráfaga 1 (debajo del setpoint)
ax_tiempo_pert1 = plt.axes([0.15, 0.20, 0.70, 0.015], facecolor='mistyrose')
ax_duracion_pert1 = plt.axes([0.15, 0.18, 0.70, 0.015], facecolor='mistyrose')
ax_intensidad_pert1 = plt.axes([0.15, 0.16, 0.70, 0.015], facecolor='mistyrose')

s_tiempo_pert1 = Slider(ax_tiempo_pert1, 'Ráfaga 1 - T.Inicio (s)', 0.0, 25.0, valinit=5.0, valstep=1.0)
s_duracion_pert1 = Slider(ax_duracion_pert1, 'Ráfaga 1 - Duración (s)', 1.0, 15.0, valinit=3.0, valstep=1.0)
s_intensidad_pert1 = Slider(ax_intensidad_pert1, 'Ráfaga 1 - Intensidad', 50.0, 500.0, valinit=150.0, valstep=10.0)

# Sliders de Perturbación - Ráfaga 2 (debajo de Ráfaga 1)
ax_tiempo_pert2 = plt.axes([0.15, 0.13, 0.70, 0.015], facecolor='lightblue')
ax_duracion_pert2 = plt.axes([0.15, 0.11, 0.70, 0.015], facecolor='lightblue')
ax_intensidad_pert2 = plt.axes([0.15, 0.09, 0.70, 0.015], facecolor='lightblue')

s_tiempo_pert2 = Slider(ax_tiempo_pert2, 'Ráfaga 2 - T.Inicio (s)', 0.0, 25.0, valinit=15.0, valstep=1.0)
s_duracion_pert2 = Slider(ax_duracion_pert2, 'Ráfaga 2 - Duración (s)', 1.0, 15.0, valinit=3.0, valstep=1.0)
s_intensidad_pert2 = Slider(ax_intensidad_pert2, 'Ráfaga 2 - Intensidad', 50.0, 500.0, valinit=100.0, valstep=10.0)

# Función para actualizar setpoint desde TextBox
def update_setpoint(text):
    try:
        valor = float(text)
        if valor > 0:  # Validar que sea positivo
            sim.setpoint = valor
            update(None)
    except ValueError:
        pass  # Ignorar si no es un número válido

# Conectar callbacks
s_kp.on_changed(update)
s_kd.on_changed(update)
tbox_setpoint.on_submit(update_setpoint)

# Ráfaga 1
s_tiempo_pert1.on_changed(update)
s_duracion_pert1.on_changed(update)
s_intensidad_pert1.on_changed(update)
# Ráfaga 2
s_tiempo_pert2.on_changed(update)
s_duracion_pert2.on_changed(update)
s_intensidad_pert2.on_changed(update)

# Botón toggle para selección de modo (esquina superior derecha)
btn_ax = plt.axes([0.73, 0.95, 0.24, 0.04])
toggle_btn = Button(btn_ax, 'Cambiar a modo DoS ⟳', color='lightblue', hovercolor='skyblue')

def toggle_scenario(event):
    # Alternar entre modos
    if sim.escenario == 'Rafagas':
        sim.escenario = 'DoS'
        toggle_btn.label.set_text('Cambiar a modo Ráfagas ⟳')
        toggle_btn.color = 'lightcoral'
        toggle_btn.ax.set_facecolor('lightcoral')
        # Ocultar controles de Ráfaga 2
        ax_tiempo_pert2.set_visible(False)
        ax_duracion_pert2.set_visible(False)
        ax_intensidad_pert2.set_visible(False)
    else:
        sim.escenario = 'Rafagas'
        toggle_btn.label.set_text('Cambiar a modo DoS ⟳')
        toggle_btn.color = 'lightblue'
        toggle_btn.ax.set_facecolor('lightblue')
        # Mostrar controles de Ráfaga 2
        ax_tiempo_pert2.set_visible(True)
        ax_duracion_pert2.set_visible(True)
        ax_intensidad_pert2.set_visible(True)
    
    update(None)

toggle_btn.on_clicked(toggle_scenario)

# Ejecución inicial
update(None)
plt.show()