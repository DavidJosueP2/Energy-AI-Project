# GUI y Gráficos

## Objetivo de esta sección

Esta sección explica cómo la interfaz gráfica recibe resultados, cómo actualiza las pestañas y cómo se construyen los gráficos a partir del `DataFrame` de simulación.

## Archivo principal

- `app/ui/gui.py`

Archivos de apoyo:

- `app/visualization/plots.py`
- `app/visualization/dashboard.py`
- `app/visualization/report_export.py`

## Papel de la GUI

La interfaz gráfica no toma decisiones de control. Su función es:

- recoger parámetros del usuario;
- lanzar simulaciones;
- lanzar optimización genética;
- mostrar inferencia manual;
- presentar resultados;
- construir gráficos y comparaciones.

## Flujo desde el botón hasta el gráfico

### Simulación base

1. El usuario presiona el botón de ejecutar control difuso base.
2. `gui.py` ejecuta `_on_run_base()`.
3. `_on_run_base()` actualiza la configuración desde la interfaz.
4. `_on_run_base()` reconstruye el controlador difuso.
5. `_on_run_base()` crea un `SimulationWorker`.
6. El worker ejecuta `Simulator.run(...)`.
7. El worker devuelve `result` y `metrics`.
8. La GUI recibe esos datos en `_on_base_finished(...)`.
9. `_on_base_finished(...)` guarda:
   - `self.base_result`
   - `self.base_metrics`
10. Luego llama a `_update_all_plots()`.

### Optimización genética

1. El usuario presiona optimización genética.
2. La GUI crea un worker de optimización.
3. El worker ejecuta el GA.
4. El GA devuelve un controlador optimizado.
5. La GUI vuelve a simular con ese controlador.
6. La GUI guarda:
   - `self.optimized_result`
   - `self.optimized_metrics`
7. La GUI vuelve a llamar `_update_all_plots()`.

## Qué hace `_update_all_plots()`

`_update_all_plots()` es el centro de actualización visual.

Responsabilidades:

- tomar `self.base_result.data`;
- tomar `self.optimized_result.data` si existe;
- construir figuras para cada pestaña;
- actualizar cada canvas;
- regenerar comparaciones base vs optimizado.

## Cómo se construye un gráfico de temperatura

En esencia, la GUI toma:

- `df_base = self.base_result.data`
- `df_opt = self.optimized_result.data`

Luego llama a una función auxiliar como `_create_comparative_plot(...)`.

Ese método usa columnas del `DataFrame`, por ejemplo:

- `time_hours`
- `temperature_indoor`
- `temperature_outdoor`

Si aplica, también dibuja:

- temperatura objetivo;
- banda de confort;
- curva base;
- curva optimizada.

## Qué representa cada gráfico típico

### Temperatura

Representa la evolución temporal de la temperatura del sistema.

En HVAC:

- temperatura interior;
- temperatura exterior;
- objetivo;
- banda de confort.

En refrigerador:

- temperatura interna del refrigerador;
- referencia u objetivo;
- en algunos paneles también puede aparecer temperatura ambiente como contexto.

### Nivel de control

Representa la salida defuzzificada del controlador.

Ejemplo:

- `0%` significa control mínimo o casi apagado;
- `80%` significa acción fuerte de enfriamiento o climatización.

### Consumo

Representa el consumo energético temporal.

Puede incluir:

- consumo del dispositivo;
- consumo total;
- comparación base vs optimizado.

### Costo acumulado

Representa cuánto dinero se va acumulando a lo largo del tiempo según:

- tarifa;
- consumo;
- acción de control.

### Confort

Representa qué tan cerca está el sistema del rango objetivo definido.

## Qué hace `plots.py`

`app/visualization/plots.py` contiene utilidades de generación de figuras más reutilizables.

Sirve para:

- encapsular estilo visual;
- construir comparaciones;
- usar rellenos cuando ayudan a resaltar diferencias;
- evitar duplicación de código gráfico.

## Qué hace `dashboard.py`

`app/visualization/dashboard.py` organiza paneles más compuestos, donde varias señales se muestran juntas para comparación rápida.

Esto es útil para:

- vista global del comportamiento;
- comparación base vs optimizado;
- presentación tipo dashboard para defensa.

## Qué hace `report_export.py`

`app/visualization/report_export.py` toma resultados y métricas para generar salidas exportables, por ejemplo HTML.

Ese reporte incluye:

- tablas métricas;
- figuras;
- comparaciones;
- resumen del sistema.

## Cómo imprimir en consola lo que usa un gráfico

Si el objetivo es defender el origen de un gráfico, lo correcto es imprimir las columnas que lo alimentan.

Ejemplo para temperatura desde `gui.py`:

```python
print(self.base_result.data[['time_hours', 'device_temperature', 'temperature_outdoor']].head(15))
```

Ejemplo para costo:

```python
print(self.base_result.data[['time_hours', 'hourly_cost', 'cumulative_cost']].head(15))
```

Ejemplo para consumo:

```python
print(self.base_result.data[['time_hours', 'device_consumption_kw', 'total_consumption_kw']].head(15))
```

## Explicación correcta para defensa

Una forma precisa de explicarlo es:

- “La simulación produce un DataFrame completo.”
- “La GUI toma ese DataFrame.”
- “Cada gráfico usa columnas específicas del DataFrame.”
- “Por eso puedo imprimir en consola exactamente los datos que alimentan cada figura.”

## Idea importante

Los gráficos son la última etapa del flujo, no la primera.

Primero ocurre:

1. inferencia difusa;
2. simulación temporal;
3. construcción del `DataFrame`;
4. cálculo de métricas;
5. visualización.

Esa secuencia debe quedar clara al momento de defender el proyecto.
