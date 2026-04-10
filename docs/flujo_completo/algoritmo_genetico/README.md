# Algoritmo Genético

## Objetivo de esta sección

Esta sección documenta el papel real del algoritmo genético dentro del proyecto.

Su función no es reemplazar la lógica difusa ni convertirse en el centro del sistema.
Su rol correcto es optimizar parámetros del controlador difuso.

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
- `app/genetic/optimizer.py`
- `app/genetic/operators.py`

También dependen directamente de:

- `app/fuzzy/controller.py`
- `app/fuzzy/rules.py`
- `app/simulation/devices.py`
- `app/simulation/simulator.py`
- `app/simulation/metrics.py`

## Documentación complementaria de operadores

Para una explicación más detallada y formal de los operadores usados en el proyecto, revisa:

- [Operadores Genéticos](./operadores/README.md)
- [Cruce BLX-α](./operadores/cruce_blx_alpha.md)
- [Mutación Gaussiana](./operadores/mutacion_gaussiana.md)
- [Elitismo](./operadores/elitismo.md)

## Qué optimiza el algoritmo genético

El GA optimiza parámetros de membresía del sistema difuso.

Ejemplos de parámetros optimizables:

- centros;
- cortes;
- anchuras;
- posiciones relativas de funciones de pertenencia;
- forma de las funciones de entrada y salida.

No optimiza:

- la simulación completa como caja negra;
- reglas completamente arbitrarias;
- decisiones de control directas por fuera del sistema difuso;
- la temperatura objetivo;
- el escenario ambiental.

## Qué representa un cromosoma

`app/genetic/chromosome.py` representa un individuo del GA.

Un cromosoma codifica una versión particular del controlador difuso, específicamente los parámetros optimizables de sus funciones de pertenencia.

Conceptualmente:

- cada gen corresponde a un parámetro del controlador;
- el conjunto de genes define una variante del sistema difuso;
- al decodificarse, esos genes reconstruyen membresías ajustadas.

### Ejemplo conceptual

Si una función triangular está definida por:

```text
alta = [1.5, 4.5, 7.5]
```

entonces esos tres números son tres genes del cromosoma.

Si el GA propone:

```text
alta = [1.2, 3.9, 6.8]
```

la etiqueta lingüística sigue siendo `alta`, pero la región del universo donde esa etiqueta se activa cambia.

Eso modifica la sensibilidad del controlador difuso sin cambiar sus reglas.

## Qué genes existen realmente

El codificador recorre:

- todas las variables de entrada del dispositivo;
- todos los conjuntos difusos de cada variable;
- todos los parámetros de cada conjunto;
- y también la variable de salida.

Para `HVAC`, hoy se optimizan estas familias:

- `temp_error`
- `humidity`
- `occupancy`
- `tariff`
- `control_output`

Si cada conjunto triangular tiene 3 parámetros, entonces el espacio de búsqueda es continuo y de varias decenas de dimensiones.

En la configuración HVAC actual:

- `temp_error`: 4 conjuntos x 3 parámetros = 12 genes
- `humidity`: 3 conjuntos x 3 parámetros = 9 genes
- `occupancy`: 4 conjuntos x 3 parámetros = 12 genes
- `tariff`: 3 conjuntos x 3 parámetros = 9 genes
- `control_output`: 5 conjuntos x 3 parámetros = 15 genes

Total aproximado:

- `57` genes reales

Por tanto, el GA no busca sobre una sola variable, sino sobre una configuración completa del sistema difuso.

## Espacio de búsqueda

El espacio de búsqueda está formado por todos los cromosomas posibles que representan funciones de pertenencia válidas.

No es un espacio libre total. Está restringido por:

- cotas mínimas y máximas por variable;
- orden de parámetros, por ejemplo `a <= b <= c`;
- cobertura del universo;
- continuidad y solape entre conjuntos vecinos;
- alcance de los extremos del universo por el primer y último conjunto.

Estas restricciones se aplican en `repair()` dentro de `app/genetic/chromosome.py`.

La interpretación correcta es:

- el cromosoma vive en un espacio continuo de alta dimensión;
- pero solo son válidos los puntos que generan familias de membresía coherentes.

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
11. Se repara cada cromosoma para mantener validez estructural.
12. Se construye una nueva generación.
13. Se conserva el mejor candidato factible encontrado.
14. Al final, ese mejor candidato define el controlador optimizado.

## Flujo real en código

### 1. Inicialización

En `app/genetic/optimizer.py`:

- se crea `ChromosomeEncoder(controller=base_controller)`;
- se crea `FitnessEvaluator(config, base_controller)`;
- se inicializa `rng = np.random.RandomState(random_seed)`.

### 2. Construcción de la población inicial

En `_init_population()`:

- el individuo `0` es exactamente el controlador difuso base;
- los demás individuos son perturbaciones aleatorias del cromosoma base.

Esto garantiza que el baseline siempre esté presente dentro de la población.

### 3. Evaluación inicial

Cada individuo se evalúa con `FitnessEvaluator.evaluate_population(population)`.

Eso no es una estimación abstracta. Cada individuo activa una simulación completa.

### 4. Bucle generacional

En cada generación:

- se seleccionan padres por torneo;
- se cruzan con BLX-α con cierta probabilidad;
- se mutan con ruido gaussiano;
- se reparan restricciones;
- se evalúa la nueva población;
- se aplica elitismo;
- se registran estadísticas.

### 5. Resultado final

El resultado final incluye:

- `best_chromosome`
- `best_fitness`
- historial por generación
- número total de evaluaciones
- tiempo total

Luego `decode_best(...)` reconstruye el controlador difuso optimizado.

## Evaluación de individuos

`app/genetic/fitness.py` coordina la evaluación de cada individuo.

La evaluación no es abstracta. Cada individuo se prueba contra la simulación real del sistema.

Eso significa que la evaluación considera:

- comportamiento temporal;
- consumo;
- costo;
- nivel de confort;
- desviación respecto al objetivo;
- estabilidad o suavidad del control;
- y otras penalizaciones definidas en métricas y fitness.

## Simulación de referencia

Antes de evaluar candidatos, el `FitnessEvaluator` construye una referencia ejecutando una simulación completa con el controlador base.

Eso produce:

- métricas base;
- un piso de confort aceptable;
- una referencia contra la cual se comparan ahorro de costo, energía y calidad térmica.

Esa referencia evita optimizar “en el vacío”.

## Función de fitness

`app/genetic/fitness.py` define cómo se juzga si un candidato es mejor o peor.

La idea actual del fitness es multiobjetivo, pero expresada como una puntuación escalar para el GA.

El criterio buscado es:

- mantener o no degradar el confort base de forma relevante;
- reducir costo;
- reducir energía;
- evitar picos innecesarios;
- evitar comportamiento inestable o físicamente poco razonable.

### Componentes del score

El `score` actual combina:

- `cost_saving`
- `energy_saving`
- `peak_saving`
- `variability_saving`
- una bonificación muy pequeña por mejora de confort
- penalizaciones por:
  - caída de confort;
  - empeoramiento de desviación media;
  - empeoramiento de desviación máxima

La lógica central es:

- el ahorro en costo pesa más que el ahorro en energía;
- el confort no puede empeorar de forma importante;
- el sistema no debería “ganar” solo por ahorrar energía si empeora demasiado la calidad de control.

## Qué significa “mejor” en este proyecto

Una solución optimizada debe tender a:

- ser igual o más confortable;
- ser igual o más económica;
- consumir igual o menos energía;
- seguir siendo interpretable como sistema difuso.

Por diseño, una solución no debería ser aceptada solo por mejorar confort si aumenta demasiado el costo o el consumo.

## Mejor candidato factible

El diseño actual no solo guarda el mejor score numérico instantáneo.

También conserva el mejor candidato factible observado durante toda la búsqueda.

Un candidato factible es uno que cumple simultáneamente:

- confort al menos igual al del controlador base;
- costo no mayor que el del controlador base;
- energía no mayor que la del controlador base.

Esto es importante porque:

- evita perder soluciones equilibradas;
- evita que una generación posterior sobrescriba una buena solución con otra menos defendible;
- mejora la coherencia entre lo que se muestra en GUI y lo que realmente conviene presentar.

## Operadores genéticos

Los operadores reales están en `app/genetic/operators.py`.

Esta sección resume su función. La explicación formal completa está separada en los documentos específicos de la carpeta `operadores`.

### Selección por torneo

La selección por torneo hace esto:

1. elige aleatoriamente `k` individuos;
2. compara sus fitness;
3. devuelve el mejor de ese grupo.

Ventaja:

- favorece individuos buenos;
- pero no exige ordenar toda la población;
- y mantiene cierta presión selectiva sin volver la evolución totalmente determinista.

### Cruce BLX-α

El cruce implementado es `BLX-alpha` para cromosomas reales.

Su objetivo es mezclar genes continuos de dos padres y permitir exploración alrededor de sus valores.

#### 1. ¿Qué significa “rango de los padres”?

Para un gen `i`, cada padre tiene un valor:

- `p1(i)`
- `p2(i)`

Por ejemplo:

- `p1(i) = 18`
- `p2(i) = 22`

Entonces, para ese gen:

- `p_min(i) = min(p1(i), p2(i)) = 18`
- `p_max(i) = max(p1(i), p2(i)) = 22`

El intervalo:

```text
[18, 22]
```

es el rango natural de los padres para ese gen.

#### 2. ¿Qué representa ese rango?

Representa la región de valores que ya está presente en los padres.

Si los hijos se generaran solo dentro de ese intervalo, el algoritmo simplemente mezclaría información conocida.

#### 3. ¿Qué significa “expandir el rango”?

BLX-α amplía ese intervalo hacia ambos lados.

Primero calcula la distancia entre padres:

```text
d(i) = p_max(i) - p_min(i)
```

Si el ejemplo es `18` y `22`:

```text
d(i) = 22 - 18 = 4
```

Luego expande con `alpha`:

```text
L(i) = p_min(i) - alpha * d(i)
U(i) = p_max(i) + alpha * d(i)
```

Si `alpha = 0.5`:

```text
L(i) = 18 - 0.5 * 4 = 16
U(i) = 22 + 0.5 * 4 = 24
```

Entonces el intervalo pasa de:

```text
[18, 22]
```

a:

```text
[16, 24]
```

#### 4. ¿Por qué se expande?

Porque si solo se generaran hijos estrictamente entre los padres, el algoritmo sería demasiado conservador.

La expansión permite:

- valores ligeramente menores que ambos padres;
- valores ligeramente mayores que ambos padres;
- exploración local alrededor de soluciones prometedoras.

Sin expansión:

- el GA mezcla lo que ya tiene.

Con expansión:

- el GA puede descubrir variantes nuevas cercanas.

#### 5. Fórmula completa del BLX-α

Para cada gen `i`:

1. identificar extremos:

```text
p_min(i) = min(p1(i), p2(i))
p_max(i) = max(p1(i), p2(i))
```

2. calcular distancia:

```text
d(i) = p_max(i) - p_min(i)
```

3. expandir intervalo:

```text
L(i) = p_min(i) - alpha * d(i)
U(i) = p_max(i) + alpha * d(i)
```

4. generar hijos:

```text
h1(i) ~ U(L(i), U(i))
h2(i) ~ U(L(i), U(i))
```

#### 6. ¿Qué significa “uniforme”?

Significa que cualquier valor dentro del intervalo tiene la misma probabilidad relativa de ser escogido.

No:

- se copia exactamente al padre;
- no se toma un promedio fijo;
- no se usa una distribución sesgada hacia el centro.

Sí:

- se muestrea aleatoriamente dentro del intervalo expandido.

#### 7. Ejemplo con un gen

Supón:

- `p1(i) = 10`
- `p2(i) = 20`
- `alpha = 0.5`

Entonces:

```text
p_min(i) = 10
p_max(i) = 20
d(i) = 10
L(i) = 5
U(i) = 25
```

Los hijos se generan así:

```text
h1(i) ~ U(5, 25)
h2(i) ~ U(5, 25)
```

Pueden salir, por ejemplo:

- `h1(i) = 8.4`
- `h2(i) = 22.1`

o:

- `h1(i) = 14.7`
- `h2(i) = 24.3`

Observación importante:

- un hijo puede quedar fuera del intervalo original de los padres;
- eso ocurre precisamente por la expansión controlada del rango.

#### 8. Ejemplo con varios genes

Si:

```text
p1 = (10, 30, 0.4)
p2 = (20, 50, 0.8)
```

BLX-α trabaja gen por gen:

- para el gen 1 construye su intervalo expandido;
- para el gen 2 construye otro;
- para el gen 3 construye otro.

Luego muestrea cada componente por separado y arma el hijo completo:

```text
h1 = (h1(1), h1(2), h1(3))
h2 = (h2(1), h2(2), h2(3))
```

#### 9. Relación con el código real

El código implementa exactamente esto:

```python
p_min = min(parent1[i], parent2[i])
p_max = max(parent1[i], parent2[i])
d = p_max - p_min

low = p_min - alpha * d
high = p_max + alpha * d

child1[i] = rng.uniform(low, high)
child2[i] = rng.uniform(low, high)
```

Eso ocurre en `blx_alpha_crossover(...)` de `app/genetic/operators.py`.

#### 10. ¿Qué pasa si `alpha = 0`?

No hay expansión.

Entonces:

```text
L(i) = p_min(i)
U(i) = p_max(i)
```

Los hijos solo salen entre los padres.

#### 11. ¿Qué pasa si `alpha > 0`?

Sí hay expansión.

Con `alpha = 0.5`, dos padres `10` y `20` producen:

```text
h(i) ~ U(5, 25)
```

Eso mejora la exploración del espacio de búsqueda.

#### 12. Forma breve para exponer

Una forma compacta y correcta de explicarlo es:

“En BLX-α, para cada gen se toma el mínimo y el máximo de los padres. Ese intervalo base se expande en ambos extremos con el parámetro α. Luego los hijos se generan aleatoriamente con distribución uniforme dentro del intervalo expandido. Esto permite combinar información existente y, al mismo tiempo, explorar soluciones cercanas nuevas.”

### Mutación gaussiana

La mutación trabaja gen por gen.

Para cada gen:

- con probabilidad `mutation_prob`;
- se suma ruido gaussiano `N(0, sigma_escalada)`.

En el código:

- la `sigma` se escala con el rango permitido de ese gen;
- por eso no se muta igual un gen de `tariff` en `[0,1]` que un gen de `control_output` en `[0,100]`.

Ventaja:

- mantiene una perturbación proporcional al dominio real del parámetro.

### Reparación

Después de cruce y mutación, un cromosoma puede quedar inválido.

Por eso `repair()`:

- recorta cada gen a sus límites;
- ordena parámetros internos;
- fuerza separaciones mínimas;
- mantiene solape entre conjuntos vecinos;
- evita huecos sin cobertura;
- hace que el primer y el último conjunto alcancen los extremos del universo.

Esto es crucial porque el GA trabaja sobre parámetros de funciones difusas, y no cualquier vector real representa una familia de membresía válida.

### Elitismo

El elitismo toma los mejores individuos de la generación anterior y reemplaza con ellos a los peores de la nueva generación.

Sirve para:

- evitar perder soluciones buenas ya encontradas;
- estabilizar la evolución;
- mejorar convergencia.

## Relación entre GA y lógica difusa

La relación correcta es:

- la lógica difusa decide;
- el GA ajusta la forma del sistema difuso;
- la simulación valida el resultado;
- la GUI compara antes y después.

Si el proyecto se defiende correctamente, debe quedar claro que:

- el conocimiento experto está en las reglas y variables difusas;
- la optimización genética solo afina ese conocimiento.

## Coste computacional y tiempo de ejecución

Una simulación base y una optimización GA no tienen el mismo costo.

Una simulación base de `72` horas ejecuta una sola trayectoria temporal.

Pero el GA evalúa muchos individuos, y cada individuo ejecuta una simulación completa.

Si:

- población = `60`
- generaciones = `50`

entonces el número aproximado de simulaciones es:

- referencia base: `1`
- población inicial: `60`
- generaciones: `50 x 60 = 3000`

Total:

- `3061` simulaciones completas

Si el horizonte es `72` horas, entonces el GA está ejecutando más de `220000` pasos de control/simulación.

Por eso una corrida GA tarda varios minutos y una simulación difusa simple tarda mucho menos.

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
- “El cruce BLX-α combina genes continuos de los padres y expande moderadamente el rango para explorar nuevas soluciones.”
- “La mutación gaussiana introduce variación controlada y la reparación garantiza que las funciones de pertenencia sigan siendo válidas.”
