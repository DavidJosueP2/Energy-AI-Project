# Flujo Completo del Proyecto

Esta carpeta documenta el flujo real del proyecto para defensa académica, mantenimiento y extensión posterior. El objetivo es que cualquier persona pueda entender:

- cómo entra un dato desde la interfaz;
- cómo ese dato llega al sistema difuso;
- cómo se ejecuta la simulación temporal;
- cómo se construye el `DataFrame` de resultados;
- cómo se dibujan los gráficos;
- cómo actúa el algoritmo genético;
- y cómo se comparan el sistema base y el optimizado.

## Mapa de documentación

- [Arquitectura General](./arquitectura/README.md)
- [Lógica Difusa](./logica_difusa/README.md)
- [Algoritmo Genético](./algoritmo_genetico/README.md)
- [Simulación y DataFrame](./simulacion_y_dataframe/README.md)
- [GUI y Gráficos](./gui_y_graficos/README.md)

## Flujo global resumido

1. La aplicación inicia desde `run.py`.
2. `app/main.py` decide si corre GUI o CLI.
3. La GUI en `app/ui/gui.py` construye la ventana principal, carga parámetros y crea el controlador difuso.
4. Cuando el usuario ejecuta la simulación base, se crea un `SimulationWorker`.
5. El worker llama a `Simulator.run(...)` en `app/simulation/simulator.py`.
6. El simulador genera el escenario temporal, consulta el controlador difuso en cada instante y construye una lista de registros.
7. Esa lista se convierte en un `pandas.DataFrame`.
8. Ese `DataFrame` queda encapsulado dentro de `SimulationResult.data`.
9. La GUI recibe `SimulationResult` y actualiza todas las pestañas y gráficos.
10. Si se ejecuta optimización genética, el GA ajusta parámetros de membresía del sistema difuso y repite simulaciones para evaluar candidatos.
11. Finalmente, la aplicación compara base vs optimizado con las mismas métricas, los mismos gráficos y la misma estructura de resultados.

## Idea central del proyecto

La arquitectura está diseñada con esta jerarquía conceptual:

- Capa principal: sistema de inferencia difusa interpretable.
- Capa secundaria: simulación temporal del entorno residencial.
- Capa terciaria: algoritmo genético como optimizador paramétrico.

Esto significa que:

- la lógica difusa toma las decisiones de control;
- la simulación le da contexto temporal y físico;
- el algoritmo genético no reemplaza el control, solo mejora sus parámetros.

## Cómo usar esta carpeta

Si quieres explicar el proyecto completo, empieza por:

- [Arquitectura General](./arquitectura/README.md)

Si quieres defender específicamente la parte “inteligente”, revisa:

- [Lógica Difusa](./logica_difusa/README.md)
- [Algoritmo Genético](./algoritmo_genetico/README.md)

Si quieres mostrar de dónde salen los gráficos o el `DataFrame`, revisa:

- [Simulación y DataFrame](./simulacion_y_dataframe/README.md)
- [GUI y Gráficos](./gui_y_graficos/README.md)
