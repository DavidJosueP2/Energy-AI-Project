# Lógica Difusa

## Objetivo de esta sección

Esta sección documenta el “cerebro” principal del proyecto: el sistema de inferencia difusa. Aquí se explica cómo se representan las variables lingüísticas, cómo se activan las reglas y cómo se obtiene una salida de control interpretable.

## Papel de la lógica difusa en el proyecto

La lógica difusa es el componente central del sistema. No es un módulo accesorio. Su función es transformar condiciones del entorno y del dispositivo en una decisión de control interpretable.

El sistema debe poder responder preguntas del tipo:

- si la temperatura es alta y la tarifa es cara, ¿qué nivel de control conviene?;
- si el refrigerador tiene temperatura interna alta y la puerta se abrió muchas veces, ¿cuánto enfriamiento aplicar?;
- si la vivienda está vacía y la temperatura está cerca del objetivo, ¿es razonable reducir climatización?

## Archivos principales

- `app/fuzzy/controller.py`
- `app/fuzzy/device_specs.py`
- `app/fuzzy/rules.py`
- `app/fuzzy/membership.py`
- `app/fuzzy/inference.py`
- `app/fuzzy/linguistic.py`

## Qué define `device_specs.py`

`app/fuzzy/device_specs.py` define la especificación difusa de cada dispositivo.

Cada especificación contiene:

- nombre del dispositivo;
- variables de entrada;
- variable de salida;
- universos de discurso;
- etiquetas lingüísticas;
- funciones de pertenencia;
- reglas aplicables;
- parámetros optimizables por el algoritmo genético;
- textos de apoyo para interpretación en GUI.

Actualmente existen al menos dos dispositivos:

- HVAC o climatización residencial;
- refrigerador.

## Variables difusas por dispositivo

### HVAC

Entradas principales:

- error térmico o temperatura interior respecto al objetivo;
- humedad;
- ocupación;
- tarifa eléctrica.

Salida:

- nivel de climatización o intensidad de control.

### Refrigerador

Entradas principales:

- temperatura interna del refrigerador;
- frecuencia de apertura;
- carga interna;
- tarifa eléctrica.

Salida:

- nivel de enfriamiento o potencia relativa del refrigerador.

## Qué hace `membership.py`

`app/fuzzy/membership.py` implementa las funciones de pertenencia.

Se usan funciones:

- triangulares;
- trapezoidales;
- o una combinación de ambas.

Estas funciones permiten mapear un valor numérico a grados de pertenencia.

Ejemplo conceptual:

- una temperatura de `29 C` puede pertenecer a:
  - `alta` con grado `0.8`
  - `muy_alta` con grado `0.2`

Eso evita decisiones rígidas tipo “todo o nada”.

## Qué hace `rules.py`

`app/fuzzy/rules.py` contiene la base de reglas difusas.

Ejemplos conceptuales:

- si temperatura es alta y humedad es alta, entonces climatización es alta;
- si temperatura es confortable y tarifa es cara, entonces climatización es baja;
- si temperatura interna del refrigerador es alta y aperturas son altas, entonces enfriamiento es muy alto.

Cada regla está escrita con estructura interpretable y trazable.

## Qué hace `controller.py`

`app/fuzzy/controller.py` coordina todo el proceso.

Responsabilidades principales:

- cargar la especificación del dispositivo;
- construir las variables difusas;
- evaluar entradas numéricas o lingüísticas;
- devolver salida defuzzificada;
- generar una traza detallada de inferencia;
- ofrecer una función de control lista para la simulación.

## Proceso interno completo de inferencia

El proceso que ejecuta el controlador es:

1. Recibe entradas.
2. Fuzzifica cada entrada.
3. Evalúa las premisas de cada regla.
4. Calcula la fuerza de activación de cada regla.
5. Recorta o implica los conjuntos difusos de salida.
6. Agrega todos los consecuentes activados.
7. Aplica desfuzzificación por centroide.
8. Determina etiqueta dominante de salida.
9. Construye explicación textual y traza de inferencia.

## Fuzzificación

Fuzzificar significa convertir una entrada numérica en grados de pertenencia.

Ejemplo conceptual:

- temperatura interna: `27 C`
- pertenencia:
  - `confortable = 0.0`
  - `alta = 0.7`
  - `muy_alta = 0.3`

Esto permite representar transiciones suaves y decisiones progresivas.

## Evaluación de reglas

Cada regla combina antecedentes usando operadores difusos.

En un esquema Mamdani típico:

- `AND` suele evaluarse como mínimo;
- `OR` suele evaluarse como máximo.

La fuerza de disparo de la regla depende de los grados de pertenencia de sus antecedentes.

Ejemplo:

- temperatura alta = `0.7`
- humedad alta = `0.6`
- entonces la fuerza de la regla:
  - `min(0.7, 0.6) = 0.6`

## Agregación y salida difusa

Cada regla activa parcialmente una etiqueta de salida:

- muy baja;
- baja;
- media;
- alta;
- muy alta.

La salida agregada es la combinación de todos esos consecuentes recortados.

No se elige una sola regla. Se combinan varias reglas activadas al mismo tiempo.

## Desfuzzificación

El proyecto usa desfuzzificación por centroide.

Esto significa:

- se calcula el centro de masa del conjunto difuso agregado;
- ese centro se devuelve como valor numérico final.

Ejemplo:

- salida lingüística dominante: `alta`
- valor numérico: `76%`

Eso permite tener:

- interpretabilidad lingüística;
- aplicabilidad numérica para el simulador.

## Salida doble: lingüística y numérica

El sistema está diseñado para mostrar:

- una salida lingüística dominante;
- un valor crisp o defuzzificado.

Ejemplo:

- climatización alta, `76%`;
- enfriamiento medio, `42%`.

Esto es importante para la defensa académica porque muestra que el sistema:

- no es caja negra;
- es interpretable;
- y sigue siendo operativo en simulación.

## Inferencia manual y simulación

El sistema soporta dos modos:

### Modo manual

El usuario selecciona etiquetas lingüísticas desde la GUI. El sistema responde con:

- activaciones;
- reglas disparadas;
- salida agregada;
- centroide;
- etiqueta dominante.

### Modo simulación temporal

La simulación entrega valores numéricos en cada hora. El controlador:

- fuzzifica internamente;
- evalúa reglas;
- devuelve la acción de control;
- registra resultados para gráficos y análisis posterior.

## Qué debe poder defenderse

Frente a un docente, esta parte debe explicarse así:

- las variables lingüísticas están definidas explícitamente;
- sus funciones de pertenencia son visibles;
- las reglas son interpretables;
- la salida resulta de agregación de reglas;
- el valor final se obtiene por centroide;
- el sistema decide el control del dispositivo de forma explicable.
