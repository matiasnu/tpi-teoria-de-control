# TPI Teoría de Control: Simulación de Controlador PD para Rate Limiter

**Alumno:** Matías Ezequiel Nuñez  
**Materia:** Teoría de Control (K4572) - UTN FRBA

Este repositorio contiene la simulación y el análisis del Trabajo Práctico Integrador que modela un sistema de **Rate Limiting** como un lazo de control cerrado con **Controlador PD** (Proporcional-Derivativo).

## 🚀 Acceso Rápido - Google Colab

**Para ejecutar la simulación sin instalar nada, haga clic aquí:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matiasnu/tpi-teoria-de-control/blob/master/notebooks/simulacion_controlador.ipynb)

La simulación interactiva incluye controles deslizantes (sliders) para ajustar las ganancias Kp y Kd en tiempo real, y permite cambiar entre diferentes escenarios de carga (Ráfagas vs. Ataque DoS).

## 📋 Descripción del Sistema

A diferencia del enfoque clásico de Token Bucket (que funciona como un controlador PI), este proyecto implementa:

1. **Controlador PD:** $G_c(s) = K_p + K_d \cdot s$
2. **Actuador con Memoria:** El mecanismo de asignación de recursos (Bucket/Autoscaler) actúa como un integrador puro en el lazo directo
3. **Realimentación Unitaria:** $H(s) = 1$

### Objetivo del TPI

Validar que el sistema es estable y presenta **error estacionario nulo** ($e_{ss}=0$) gracias a la naturaleza "Tipo 1" del lazo completo, a pesar de que el controlador es PD (sin acción integral explícita).

La clave es que el **actuador tiene memoria** (acumula recursos/tokens), lo que añade un polo en el origen al lazo abierto, convirtiendo al sistema en Tipo 1.

## 🧪 Escenarios de Simulación

La simulación analiza el comportamiento del sistema bajo dos escenarios:

1. **Ráfagas de Tráfico:** Evalúa la respuesta transitoria ante picos cortos de tráfico (t=5s y t=15s)
2. **Ataque DoS Sostenido:** Comprueba la estabilidad y el error en estado estacionario ante una perturbación constante desde t=5s

## 💻 Instalación Local (Opcional)

Si prefiere ejecutar la simulación localmente en lugar de usar Google Colab:

### Opción 1: Ejecutar el Notebook Interactivo

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/matiasnu/tpi-teoria-de-control.git
   cd tpi-teoria-de-control
   ```

2. (Recomendado) Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Iniciar Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Abrir el archivo `notebooks/simulacion_controlador.ipynb` desde la interfaz de Jupyter

**Nota:** Para usar widgets interactivos en VSCode o Jupyter Lab, necesitará instalar:
```bash
pip install ipympl
```
Y usar `%matplotlib widget` en lugar de `%matplotlib inline`.

### Opción 2: Ejecutar el Simulador Standalone

Para ejecutar el simulador gráfico sin Jupyter:

```bash
python sim/controlador_pd.py
```

Esto abrirá una ventana interactiva con matplotlib donde podrá ajustar los parámetros Kp y Kd mediante sliders.

## 📂 Estructura del Proyecto

```
tpi-teoria-de-control/
│
├── notebooks/
│   └── simulacion_controlador.ipynb    # Notebook interactivo (compatible con Colab)
│
├── sim/
│   ├── __init__.py
│   └── controlador_pd.py               # Simulador standalone con matplotlib
│
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Este archivo
```

## 🎮 Uso de la Simulación Interactiva

### Controles Disponibles

- **Slider Kp (Ganancia Proporcional):** Rango 0.0 - 5.0
  - ↑ Kp: Respuesta más rápida, pero puede causar sobrepicos (overshoot)
  - ↓ Kp: Respuesta más lenta y suave

- **Slider Kd (Ganancia Derivativa):** Rango 0.0 - 5.0
  - ↑ Kd: Mayor amortiguamiento, reduce oscilaciones
  - ↓ Kd: Menor amortiguamiento

- **Selector de Escenario:**
  - **Ráfagas:** Picos de tráfico cortos en t=5s (150 req/s) y t=15s (80 req/s)
  - **DoS:** Ataque sostenido de 400 req/s desde t=5s hasta el final

### Gráficos Generados

1. **Respuesta del Sistema:** Muestra el Setpoint (θᵢ), la salida del sistema (Y) y la perturbación (D)
2. **Error:** Muestra e(t) = R(t) - Y(t)
3. **Señal de Control:** Muestra u(t), la salida del controlador PD

## 🔬 Resultados Esperados

### Escenario Ráfagas
- El sistema debe responder rápidamente a los picos
- Mayor Kd reduce las oscilaciones
- La salida debe volver al setpoint después de cada ráfaga

### Escenario DoS
- **Resultado clave:** El error debe converger a cero en estado estacionario (e_ss = 0)
- Esto valida que el sistema es Tipo 1 gracias a la memoria del actuador
- El sistema debe mantener estabilidad incluso bajo carga sostenida