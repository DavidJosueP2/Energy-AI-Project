# Simulación y DataFrame

## Objetivo de esta sección

Esta sección explica cómo se ejecuta la simulación temporal, cómo se construyen los resultados y por qué el `DataFrame` es la fuente directa de los gráficos.

Documentación matemática complementaria:

- [Modelo Matemático del Entorno y la Dinámica](./modelo_matematico_entorno_y_dinamica.md)
- [Parámetros Reales Bosch y Mitsubishi](./parametros_reales_bosch_mitsubishi.md)

## Archivos principales

- `app/simulation/simulator.py`
- `app/simulation/environment.py`
- `app/simulation/devices.py`
- `app/simulation/metrics.py`
- `app/simulation/scenario_generator.py`

## Qué hace la simulación

La simulación representa el comportamiento del sistema a lo largo del tiempo.

En cada instante:

- observa el entorno;
- consulta el controlador difuso;
- aplica la acción de control al dispositivo;
- actualiza el estado;
- calcula consumo y costo;
- y registra todo en una estructura tabular.

## Flujo interno de `Simulator.run(...)`

El método `run(...)` de `app/simulation/simulator.py` ejecuta estos pasos:

1. Lee la configuración activa.
2. Genera el escenario temporal.
3. Construye perfiles del entorno.
4. Inicializa el dispositivo controlado.
5. Recorre cada hora o intervalo de simulación.
6. Arma las entradas del controlador difuso.
7. Llama al controlador para obtener el nivel de control.
8. Aplica ese control al modelo dinámico del dispositivo.
9. Calcula consumo base, consumo del dispositivo, costo y métricas de confort.
10. Guarda todo en un diccionario por instante.
11. Agrega cada diccionario a una lista `records`.
12. Convierte `records` a `pandas.DataFrame`.
13. Devuelve `SimulationResult`, que encapsula ese `DataFrame`.

## Qué es `SimulationResult`

`SimulationResult` es una estructura de resultados.

Su componente más importante es:

- `SimulationResult.data`

Ese atributo contiene el `DataFrame` con todos los resultados de la simulación.

## Qué representa el DataFrame

El `DataFrame` es, en términos prácticos, una tabla tipo Excel en memoria.

Características:

- cada fila representa un instante de simulación;
- cada columna representa una variable;
- el conjunto completo representa la historia temporal del experimento.

## Ejemplos de columnas

Dependiendo del dispositivo y la configuración, el `DataFrame` puede contener columnas como:

- `time_hours`
- `temperature_indoor`
- `temperature_outdoor`
- `device_temperature`
- `target_temperature`
- `temp_error`
- `control_level`
- `device_consumption_kw`
- `base_consumption_kw`
- `total_consumption_kw`
- `tariff`
- `hourly_cost`
- `cumulative_cost`
- `comfort_index`

## Qué significa cada fila

Por ejemplo, una fila puede representar:

- hora `17`
- temperatura interior `24.2 C`
- temperatura exterior `32.8 C`
- nivel de control `48%`
- consumo total `1.31 kW`
- costo acumulado `6.84`
- confort `0.92`

Ese único registro ya puede alimentar varias gráficas a la vez.

## Cómo nace el DataFrame

En `app/simulation/simulator.py`, durante el recorrido temporal:

- se crea un diccionario `record` por cada paso;
- luego se usa `records.append(record)`;
- al final, la lista completa se convierte en `DataFrame`;
- ese `DataFrame` se entrega en `SimulationResult.data`.

## Relación entre simulación y lógica difusa

La simulación no reemplaza a la lógica difusa.

La relación correcta es:

- la lógica difusa calcula la acción;
- la simulación muestra qué ocurre cuando esa acción se aplica en el tiempo.

Esto permite evaluar si la decisión difusa realmente:

- mantiene confort;
- reduce consumo;
- reduce costo;
- y responde razonablemente ante cambios del entorno.

## Dónde están las fórmulas

La documentación narrativa de esta carpeta explica el flujo. Las fórmulas
principales del modelo están resumidas en:

- [Modelo Matemático del Entorno y la Dinámica](./modelo_matematico_entorno_y_dinamica.md)

Ahí se describen formalmente:

- temperatura exterior;
- humedad;
- radiación solar;
- tarifa normalizada;
- carga base;
- aproximación de temperatura ambiente del refrigerador;
- balance térmico del HVAC;
- balance térmico del refrigerador;
- consumo eléctrico;
- costo por paso temporal;
- e índice de confort.

## Métricas calculadas

`app/simulation/metrics.py` extrae métricas resumidas a partir del `DataFrame`.

Ejemplos:

- energía total;
- energía del dispositivo;
- costo total;
- demanda pico;
- confort porcentual;
- desviación promedio;
- desviación máxima;
- variabilidad del control.

Estas métricas son las que se usan para:

- tablas comparativas;
- reportes;
- evaluación del GA;
- indicadores del dashboard.

## Cómo imprimir el DataFrame para defensa

Si se quiere demostrar de dónde sale un gráfico, basta imprimir parte del `DataFrame`.

Ejemplo conceptual:

```python
print(self.base_result.data[['time_hours', 'temperature_indoor', 'temperature_outdoor']].head(10))
```

Eso muestra:

- el tiempo;
- la temperatura interior;
- la temperatura exterior;
- y permite justificar el origen de la gráfica de temperatura.

## Idea clave para defensa

Cuando pregunten “¿de dónde sale ese gráfico?”, la respuesta correcta es:

- “Sale de columnas del DataFrame generado por la simulación.”

No sale de valores manuales ni de un dibujo aislado.

Sale de:

1. simulación;
2. registros horarios;
3. `DataFrame`;
4. gráfico construido a partir de sus columnas.
