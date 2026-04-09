# Algoritmo Genético

## Objetivo de esta sección

Esta sección documenta el papel real del algoritmo genético dentro del proyecto. Su función no es reemplazar la lógica difusa ni convertirse en el centro del sistema. Su rol correcto es optimizar parámetros del controlador difuso.

## Idea central

El algoritmo genético trabaja sobre el controlador difuso, no en lugar del controlador difuso.

Esto significa:

- el sistema base ya funciona con lógica difusa;
- el GA toma ese sistema base;
- modifica parámetros optimizables;
- simula el comportamiento resultante;
- y selecciona mejores configuraciones.

## Archivos principales

- `app/genetic/chromosome.py`
- `app/genetic/fitness.py`
- `app/genetic/evaluation.py`
- `app/genetic/optimizer.py`

También dependen directamente de:

- `app/fuzzy/controller.py`
- `app/fuzzy/device_specs.py`
- `app/simulation/simulator.py`
- `app/simulation/metrics.py`

## Qué optimiza el algoritmo genético

El GA optimiza parámetros de membresía del sistema difuso.

Ejemplos de parámetros optimizables:

- centros;
- cortes;
- anchuras;
- posiciones relativas de funciones de pertenencia;
- algunos pesos o escalas razonables, si la especificación lo permite.

No optimiza:

- la simulación completa como caja negra;
- reglas completamente arbitrarias;
- decisiones de control directas por fuera del sistema difuso.

## Qué representa un cromosoma

`app/genetic/chromosome.py` representa un individuo del GA.

Un cromosoma codifica una versión particular del controlador difuso, específicamente los parámetros optimizables de sus funciones de pertenencia.

Conceptualmente:

- cada gen corresponde a un parámetro del controlador;
- el conjunto de genes define una variante del sistema difuso;
- al decodificarse, esos genes reconstruyen membresías ajustadas.

## Flujo detallado del GA

El proceso general es:

1. Se parte de un controlador difuso base.
2. Se identifican los parámetros optimizables.
3. Se genera una población inicial de cromosomas.
4. Cada cromosoma se decodifica para producir un controlador candidato.
5. Ese controlador candidato se evalúa ejecutando una simulación completa.
6. La simulación produce métricas.
7. Las métricas se convierten en un valor de fitness.
8. Se aplica selección.
9. Se generan hijos por cruce.
10. Se aplican mutaciones.
11. Se construye una nueva generación.
12. Se conserva el mejor candidato factible encontrado.
13. Al final, ese mejor candidato define el controlador optimizado.

## Evaluación de individuos

`app/genetic/evaluation.py` coordina la evaluación de cada individuo.

La evaluación no es abstracta. Cada individuo se prueba contra la simulación real del sistema.

Eso significa que la evaluación considera:

- comportamiento temporal;
- consumo;
- costo;
- nivel de confort;
- desviación respecto al objetivo;
- estabilidad o suavidad del control;
- y otras penalizaciones definidas en métricas y fitness.

## Función de fitness

`app/genetic/fitness.py` define cómo se juzga si un candidato es mejor o peor.

La idea actual del fitness es multiobjetivo, pero expresada como una puntuación escalar para el GA.

El criterio buscado es:

- mantener o no degradar el confort base de forma relevante;
- reducir costo;
- reducir energía;
- evitar picos innecesarios;
- evitar comportamiento inestable o físicamente poco razonable.

## Qué significa “mejor” en este proyecto

Una solución optimizada debe tender a:

- ser igual o más confortable;
- ser igual o más económica;
- consumir igual o menos energía;
- seguir siendo interpretable como sistema difuso.

Por diseño, una solución no debería ser aceptada solo por mejorar confort si aumenta demasiado el costo o el consumo.

## Selección, cruce y mutación

`app/genetic/optimizer.py` implementa la evolución de la población.

En términos generales:

- selección: elige individuos más prometedores;
- cruce: combina genes de padres distintos;
- mutación: introduce variación para explorar nuevas soluciones.

Esto permite recorrer el espacio de parámetros sin probar todas las combinaciones posibles de forma exhaustiva.

## Mejor candidato factible

Uno de los puntos importantes del diseño actual es que el optimizador no solo conserva el mejor score instantáneo de una generación. También rastrea el mejor candidato factible observado durante toda la búsqueda.

Esto es importante porque:

- evita perder soluciones equilibradas;
- evita que una última generación sobrescriba una buena solución previa con otra menos defendible;
- mejora la consistencia entre lo que muestra el fitness y lo que realmente conviene presentar en GUI.

## Relación entre GA y lógica difusa

La relación correcta es:

- la lógica difusa decide;
- el GA ajusta la forma del sistema difuso;
- la simulación valida el resultado;
- la GUI compara antes y después.

Si el proyecto se defiende correctamente, debe quedar claro que:

- el conocimiento experto está en las reglas y variables difusas;
- la optimización genética solo afina ese conocimiento.

## Qué debe observarse en GUI o reporte

El módulo de optimización debe permitir mostrar:

- evolución del fitness por generación;
- métricas base vs optimizadas;
- comparación de funciones de pertenencia antes y después;
- efecto sobre consumo, costo y confort;
- cambio en el patrón de control.

## Qué debe decirse en una defensa

Una explicación sólida sería:

- “Primero definimos un controlador difuso interpretable.”
- “Luego usamos algoritmo genético únicamente para afinar sus parámetros de membresía.”
- “Cada individuo del GA representa una variante del sistema difuso.”
- “Cada variante se evalúa ejecutando la simulación completa.”
- “La mejor solución se selecciona por su equilibrio entre confort, costo y energía.”
