# TPI Teoría de Control: Modelado de Rate Limiter (PI No Lineal)

Este repositorio contiene la simulación y el análisis para el Trabajo Práctico Integrador de la materia Teoría de Control (K4572) de la UTN FRBA.

## Descripción

El proyecto modela un sistema de *Rate Limiter* (basado en el algoritmo *Token Bucket*) como un sistema de control de lazo cerrado, fundamentado en la teoría de W. Bolton.

La tesis central del trabajo es que el algoritmo **Token Bucket** no es una simple heurística, sino un **Controlador Proporcional-Integral (PI) No Lineal**:

*   **Acción Integral (I):** El "balde" de tokens, que acumula (integra) el error entre la tasa de generación (Referencia, `R`) y la tasa de consumo (Salida, `Y`).
*   **Acción Proporcional (P):** La lógica de decisión ON/OFF (`Permitir/Rechazar`), que actúa como un control proporcional de dos posiciones (ganancia infinita) basado en el estado del integrador.
*   **Ausencia de Acción Derivativa (D):** El sistema es *reactivo* (al estado del balde) y no *predictivo* (no mide la *velocidad* del pico de tráfico).

## Simulación y Análisis

La simulación (implementada en `sim/token_bucket.py`) analiza el comportamiento del sistema bajo dos escenarios clave para validar el modelo PI:

1.  **Respuesta Transitoria a Ráfagas:** Analiza cómo el parámetro `B` (Capacidad del Bucket) y la Acción Integral gestionan picos de tráfico cortos.
2.  **Estabilidad y Error en Estado Estacionario (Ataque DoS):** Comprueba la predicción teórica de que un sistema Tipo 1 (con PI) tendrá un **Error en Estado Estacionario Nulo (`e_ss = 0`)** frente a una perturbación de escalón sostenida.

## Instalación

1.  Clonar el repositorio:bash
    git clone [https://github.com/](https://github.com/matiasnu/tpi-teoria-de-control.git)
    cd tpi-teoria-de-control
    ```

2.  (Recomendado) Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar las simulaciones y visualizar el análisis, inicie el servidor de Jupyter Notebook:

```bash
jupyter notebook
```

Luego, abra el archivo `notebooks/analisis_simulacion.ipynb` desde la interfaz de Jupyter en su navegador.
```