# Elitismo

## Objetivo

Este documento explica el operador de elitismo usado en el proyecto.

Su propósito es mostrar:

- qué significa preservar élites;
- cómo se identifican en el código;
- cómo se insertan en la nueva generación;
- y por qué este operador mejora la estabilidad del algoritmo genético.

## Contexto dentro del proyecto

El elitismo está implementado en:

- `app/genetic/operators.py`

Se aplica después de:

- selección;
- cruce;
- mutación;
- reparación;
- evaluación de la nueva población.

Esto significa que el elitismo no reemplaza la reproducción normal del GA.  
Actúa al final del ciclo generacional para evitar perder las mejores soluciones ya encontradas.

## Idea intuitiva

El elitismo dice:

“Antes de aceptar completamente la nueva generación, aseguro que algunos de los mejores individuos de la generación anterior sobrevivan”.

Sin elitismo puede ocurrir que:

- una generación produzca accidentalmente hijos peores;
- y el algoritmo pierda una solución buena encontrada antes.

Con elitismo:

- ciertas soluciones destacadas se preservan explícitamente.

## Flujo general del elitismo

La implementación hace esto:

1. identificar los mejores individuos de la población anterior;
2. identificar los peores individuos de la población nueva;
3. reemplazar a los peores nuevos por las mejores élites antiguas.

El resultado es:

- se conserva la exploración de la nueva generación;
- pero no se sacrifica completamente la calidad ya alcanzada.

## La línea clave

En el código aparece:

```python
elite_indices = np.argsort(old_fitnesses)[-elite_count:]
```

Esta línea obtiene los índices de los `elite_count` mejores individuos de la generación anterior.

## 1. Qué hace `np.argsort(old_fitnesses)`

`np.argsort(...)` no devuelve los valores ordenados.  
Devuelve los índices que ordenarían el arreglo de menor a mayor.

### Ejemplo

```python
old_fitnesses = [10, 50, 30]
np.argsort(old_fitnesses)
```

Resultado:

```python
[0, 2, 1]
```

Interpretación:

- el valor `10` está en el índice `0`;
- el valor `30` está en el índice `2`;
- el valor `50` está en el índice `1`.

Como `argsort` ordena de menor a mayor, el último índice corresponde al mejor fitness.

## 2. Qué hace `[-elite_count:]`

Después de ordenar todos los índices, se toman los últimos `elite_count`.

Como la lista está en orden ascendente:

- los últimos son los que corresponden a los fitness más altos.

Entonces:

```python
elite_indices = np.argsort(old_fitnesses)[-elite_count:]
```

significa:

- “dame los índices de las mejores élites”.

## 3. Ejemplo completo

Supón:

```python
old_fitnesses = [10, 50, 30, 80, 60]
elite_count = 2
```

Primero:

```python
np.argsort(old_fitnesses)
```

da:

```python
[0, 2, 1, 4, 3]
```

porque:

- `10` está en índice `0`;
- `30` en índice `2`;
- `50` en índice `1`;
- `60` en índice `4`;
- `80` en índice `3`.

Luego:

```python
np.argsort(old_fitnesses)[-2:]
```

da:

```python
[4, 3]
```

Eso significa que las élites son:

- índice `4` con fitness `60`;
- índice `3` con fitness `80`.

Observación:

- los índices aparecen en orden ascendente dentro del recorte final;
- pero ambos corresponden a los mejores individuos.

## 4. La otra parte importante

El código también hace:

```python
worst_indices = np.argsort(new_fitnesses)[:elite_count]
```

Esto obtiene los índices de los peores individuos de la nueva población.

Aquí el razonamiento es el inverso:

- como `argsort` devuelve de menor a mayor;
- los primeros elementos corresponden a los fitness más bajos.

## 5. Reemplazo de individuos

Después, el código hace:

```python
for elite_idx, worst_idx in zip(elite_indices, worst_indices):
    result_pop[worst_idx] = old_population[elite_idx].copy()
    result_fit[worst_idx] = old_fitnesses[elite_idx]
```

La interpretación es:

- tomar una élite antigua;
- tomar un individuo malo de la nueva generación;
- reemplazar el malo por la élite.

Así:

- la nueva generación no pierde totalmente a sus hijos;
- pero tampoco puede degradarse por completo respecto a la anterior.

## Forma matemática del operador

Sea:

- `P_old` la población anterior;
- `F_old` sus fitness;
- `P_new` la nueva población;
- `F_new` sus fitness;
- `e` el número de élites.

Entonces:

1. se obtiene el conjunto de índices de élite:

```text
E = indices de los e mayores valores de F_old
```

2. se obtiene el conjunto de peores índices de la nueva población:

```text
W = indices de los e menores valores de F_new
```

3. se reemplaza:

```text
P_new[W_j] <- P_old[E_j]
F_new[W_j] <- F_old[E_j]
```

para `j = 1, ..., e`.

## Por qué es útil

El elitismo aporta estabilidad evolutiva.

Sus ventajas son:

- evita perder buenas soluciones por azar;
- mejora la monotonicidad del mejor fitness observado;
- acelera la convergencia práctica;
- hace más defendible el resultado final.

En este proyecto eso es importante porque cada evaluación cuesta mucho:

- un individuo implica reconstruir un controlador difuso;
- correr una simulación completa;
- calcular métricas;
- y convertirlas en fitness.

Perder una buena solución ya evaluada sería costoso y poco eficiente.

## Riesgo del elitismo

El elitismo también tiene un riesgo:

- si se usa en exceso, puede volver al GA demasiado conservador.

Si el número de élites es muy grande:

- la población pierde diversidad;
- la exploración se reduce;
- y el algoritmo puede estancarse.

Por eso en práctica se usa:

- un número pequeño de élites;
- suficiente para preservar calidad, pero no tanto como para bloquear evolución.

## Relación con el proyecto

En este sistema, el elitismo no significa preservar “decisiones de control”.

Significa preservar:

- configuraciones paramétricas del sistema difuso;
- es decir, cromosomas que representan funciones de pertenencia prometedoras.

Por eso, académicamente, su interpretación correcta es:

- proteger buenas configuraciones del controlador difuso ya descubiertas.

## Forma breve para exponer

Una forma correcta y clara de explicarlo es:

“El elitismo preserva algunos de los mejores individuos de la generación anterior. Para ello, se identifican sus índices usando `argsort` sobre el vector de fitness, se localizan los peores individuos de la nueva generación y se reemplazan por esas élites. Así se evita perder soluciones buenas ya encontradas y se mejora la estabilidad del proceso evolutivo.”
