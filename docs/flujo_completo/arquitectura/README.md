# Arquitectura General

## Objetivo de esta sección

Esta documentación explica cómo está organizado el proyecto a nivel estructural y qué responsabilidad tiene cada módulo principal.

## Capas conceptuales

El proyecto está dividido en tres capas funcionales:

1. Sistema de inferencia difusa.
2. Simulación temporal del entorno y del dispositivo.
3. Optimización genética de parámetros del sistema difuso.

La lógica difusa es el núcleo del sistema. La simulación le entrega entradas dinámicas y el algoritmo genético ajusta sus funciones de pertenencia para mejorar desempeño.

## Estructura principal del proyecto

### Entrada principal

- `run.py`
- `app/main.py`

Responsabilidad:

- arrancar la aplicación;
- decidir entre modo GUI o CLI;
- crear la ventana principal o ejecutar flujo de consola.

### Interfaz gráfica

- `app/ui/gui.py`

Responsabilidad:

- construir la interfaz;
- capturar parámetros del usuario;
- lanzar simulación base;
- lanzar optimización genética;
- recibir resultados;
- construir pestañas y gráficos;
- mostrar inferencia difusa manual y resultados comparativos.

### Núcleo difuso

- `app/fuzzy/controller.py`
- `app/fuzzy/device_specs.py`
- `app/fuzzy/rules.py`
- `app/fuzzy/membership.py`
- `app/fuzzy/inference.py`
- `app/fuzzy/linguistic.py`

Responsabilidad:

- definir variables lingüísticas;
- definir universos y funciones de pertenencia;
- definir reglas;
- fuzzificar entradas;
- evaluar reglas;
- agregar salida;
- desfuzzificar por centroide;
- exponer trazabilidad de la inferencia.

### Simulación temporal

- `app/simulation/simulator.py`
- `app/simulation/environment.py`
- `app/simulation/devices.py`
- `app/simulation/metrics.py`
- `app/simulation/scenario_generator.py`

Responsabilidad:

- generar el entorno horario;
- modelar la dinámica del dispositivo;
- aplicar la acción de control;
- calcular consumo, costo y confort;
- construir el `DataFrame` final de resultados;
- calcular métricas para base y optimizado.

### Optimización genética

- `app/genetic/chromosome.py`
- `app/genetic/fitness.py`
- `app/genetic/evaluation.py`
- `app/genetic/optimizer.py`

Responsabilidad:

- codificar parámetros optimizables del controlador difuso;
- generar población inicial;
- evaluar individuos usando simulación real;
- aplicar selección, cruce y mutación;
- conservar el mejor candidato factible;
- devolver un controlador optimizado.

### Visualización y exportación

- `app/visualization/plots.py`
- `app/visualization/dashboard.py`
- `app/visualization/report_export.py`

Responsabilidad:

- generar figuras auxiliares;
- construir dashboards comparativos;
- exportar HTML y CSV.

## Flujo arquitectónico completo

1. La GUI construye la configuración.
2. La GUI crea el controlador difuso según el dispositivo elegido.
3. La GUI crea un worker para ejecutar simulación sin bloquear la ventana.
4. El worker llama al simulador.
5. El simulador consulta al controlador difuso en cada hora.
6. El simulador construye un `DataFrame` con resultados.
7. El simulador devuelve `SimulationResult`.
8. La GUI recibe `SimulationResult`.
9. La GUI actualiza gráficos, tablas y paneles.
10. Si el usuario ejecuta GA, se repite el proceso muchas veces con distintos parámetros de membresía.
11. El mejor individuo genera un nuevo controlador.
12. La GUI vuelve a simular y compara ambos sistemas.

## Principio de diseño defendible

Este proyecto debe explicarse así:

- el controlador difuso decide;
- la simulación verifica cómo se comporta esa decisión a lo largo del tiempo;
- el algoritmo genético ajusta parámetros para mejorar el controlador;
- la interfaz permite observar todo el proceso.
