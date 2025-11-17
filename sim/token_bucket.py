import numpy as np
import matplotlib.pyplot as plt


class TokenBucketSimulator:
    """
    Clase para simular un sistema de Rate Limiter (Token Bucket).

    Parámetros:
    - R_rate (float): Tasa de generación de tokens (tokens/seg). Es el Setpoint (R).
    - B_capacity (float): Capacidad máxima del bucket (tokens).
    - dt (float): Paso de tiempo de la simulación (segundos).
    """

    def __init__(self, R_rate, B_capacity, dt=0.01):
        self.R_rate = R_rate
        self.B_capacity = B_capacity
        self.dt = dt
        self.tokens = B_capacity  # El bucket comienza lleno

    def simulate(self, traffic_pattern):
        """
        Ejecuta la simulación contra un patrón de tráfico dado.

        Parámetros:
        - traffic_pattern (list or np.array): Lista de la tasa de requests
                                             entrantes en cada paso de tiempo.

        Retorna:
        - (dict): Un diccionario con los resultados de la simulación (series de tiempo).
        """
        n_steps = len(traffic_pattern)

        # Arrays para almacenar resultados
        sim_time = np.arange(0, n_steps * self.dt, self.dt)
        allowed_rate = np.zeros(n_steps)
        rejected_rate = np.zeros(n_steps)
        token_levels = np.zeros(n_steps)

        # Resetea el bucket al inicio de la simulación
        self.tokens = self.B_capacity

        for i in range(n_steps):
            # 1. Generar tokens (Acción Integral)
            # Se añaden tokens proporcionales al paso de tiempo.
            tokens_generated = self.R_rate * self.dt
            self.tokens = min(self.B_capacity, self.tokens + tokens_generated)

            # 2. Calcular requests entrantes en este paso
            incoming_requests = traffic_pattern[i] * self.dt

            # 3. Consumir tokens (Acción de Control Proporcional ON/OFF)
            if incoming_requests <= self.tokens:
                # Hay suficientes tokens (Estado ON): permitir requests y consumir
                allowed = incoming_requests
                rejected = 0
                self.tokens -= allowed
            else:
                # No hay suficientes tokens (Estado OFF): permitir solo lo que queda y rechazar
                allowed = self.tokens
                rejected = incoming_requests - allowed
                self.tokens = 0  # El bucket se vacía (saturación del integrador)

            # 4. Almacenar métricas (convertidas de nuevo a tasa por segundo)
            allowed_rate[i] = allowed / self.dt
            rejected_rate[i] = rejected / self.dt
            token_levels[i] = self.tokens

        return {
            "time": sim_time,
            "incoming_rate": traffic_pattern,
            "allowed_rate": allowed_rate,
            "rejected_rate": rejected_rate,
            "token_levels": token_levels
        }


def plot_simulation(results, title):
    """Función helper para graficar resultados con Matplotlib."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Gráfico Superior: Tasas de Requests
    ax1.plot(results["time"], results["incoming_rate"], 'orange', linestyle='--',
             label='Tasa de Llegada (Perturbación D(t))')
    ax1.plot(results["time"], results["allowed_rate"], 'b-', label='Tasa Permitida (Salida Y(t))')
    ax1.axhline(y=100.0, color='r', linestyle=':', label='Referencia R(t) = 100')  # Línea de referencia
    ax1.set_ylabel('Tasa de Requests (req/s)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Gráfico Inferior: Nivel de Tokens (Acción Integral)
    ax2.plot(results["time"], results["token_levels"], 'g-', label='Tokens en Bucket (Integral del Error)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Tokens')
    ax2.set_ylim(bottom=-5)  # Empezar en -5 para ver el cero
    ax2.axhline(y=0, color='gray', linestyle='--', label='Bucket Vacío (Estado OFF)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()