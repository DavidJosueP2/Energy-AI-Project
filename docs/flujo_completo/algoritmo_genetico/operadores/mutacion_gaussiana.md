# Mutación Gaussiana

## Objetivo

Este documento explica con detalle formal la mutación gaussiana usada en el proyecto.

La meta es dejar claro:

- cuándo un gen se muta;
- cuánto puede cambiar;
- cómo se escala ese cambio;
- y cuál es la relación entre la fórmula y el código real.

## Contexto dentro del proyecto

En este sistema, los cromosomas están formados por genes reales.

Cada gen representa un parámetro continuo de una función de pertenencia, por ejemplo:

- un extremo izquierdo `a`;
- un centro `b`;
- un extremo derecho `c`.

Por tanto, la mutación no consiste en cambiar símbolos discretos.

Consiste en perturbar valores reales de manera controlada.

## Idea intuitiva

La mutación gaussiana implementada puede resumirse así:

1. revisar cada gen de forma independiente;
2. decidir al azar si ese gen se muta o no;
3. si se muta, sumar un pequeño ruido aleatorio;
4. hacer que la magnitud de ese ruido dependa del rango permitido del gen.

La idea principal es:

- introducir novedades locales;
- pero sin destruir completamente la estructura de una buena solución.

## Forma breve del operador

Si un gen actual es `x(i)`, después de mutación puede quedar:

```text
x'(i) = x(i) + N(0, sigma_i)
```

donde:

- `N(0, sigma_i)` es ruido gaussiano centrado en `0`;
- `sigma_i` es la desviación estándar efectiva de ese gen.

## 1. Decisión de mutar o no

En el código aparece esta condición:

```python
if rng.random() < mutation_prob:
```

La interpretación es:

- se genera un número aleatorio uniforme entre `0` y `1`;
- si ese número es menor que la probabilidad de mutación, el gen se modifica;
- en caso contrario, se deja intacto.

Matemáticamente:

```text
r_i < p_m
```

donde:

- `r_i` es un número aleatorio en `[0,1]`;
- `p_m` es la probabilidad de mutación.

### Intuición

Esto implica que:

- no todos los genes cambian;
- solo algunos se alteran;
- la mutación está distribuida a lo largo del cromosoma.

## 2. Rango del gen

En el código:

```python
gene_range = gene_specs[i]['max'] - gene_specs[i]['min']
```

Eso mide el tamaño del intervalo permitido para ese gen.

### Ejemplo 1

Si:

- `min = 0`
- `max = 100`

entonces:

```text
gene_range = 100
```

### Ejemplo 2

Si:

- `min = 0`
- `max = 1`

entonces:

```text
gene_range = 1
```

### Significado

Cada gen vive en una escala distinta.

Por eso no sería correcto mutar igual:

- un gen de salida en `[0,100]`;
- y un gen de tarifa en `[0,1]`.

## 3. Escalado de la desviación estándar

En el código:

```python
scaled_sigma = sigma * (gene_range / 20.0)
```

Esta línea define qué tan fuerte será realmente la mutación para ese gen.

### Fórmula

```text
sigma_i = sigma * (R_i / 20)
```

donde:

- `sigma` es el parámetro global de mutación;
- `R_i` es el rango del gen.

### Ejemplo 1

Si:

- `sigma = 0.1`
- `gene_range = 100`

entonces:

```text
scaled_sigma = 0.1 * (100 / 20) = 0.5
```

### Ejemplo 2

Si:

- `sigma = 0.1`
- `gene_range = 1`

entonces:

```text
scaled_sigma = 0.1 * (1 / 20) = 0.005
```

### Intuición fundamental

Esto significa que la mutación es proporcional a la escala del gen.

Por tanto:

- genes grandes reciben perturbaciones más grandes;
- genes pequeños reciben perturbaciones más finas.

Esa es una decisión muy importante de diseño.

Sin este escalado:

- algunos genes mutarían demasiado;
- otros casi no cambiarían en términos relativos.

## 4. Mutación real del valor

En el código:

```python
mutated[i] += rng.normal(0, scaled_sigma)
```

La función `rng.normal(0, scaled_sigma)` genera un valor aleatorio gaussiano:

- centrado en `0`;
- con desviación estándar `scaled_sigma`.

Eso quiere decir que:

- los cambios pequeños son más probables;
- los cambios grandes son posibles, pero menos frecuentes.

### Fórmula

```text
x'(i) = x(i) + N(0, sigma_i)
```

### Ejemplo

Si:

- gen actual = `50`
- ruido generado = `+0.4`

entonces:

```text
nuevo_gen = 50.4
```

Si el ruido es `-0.3`, entonces:

```text
nuevo_gen = 49.7
```

### Significado

La mutación no reemplaza el valor.

Lo que hace es:

- desplazarlo ligeramente;
- hacia arriba o hacia abajo;
- con magnitud variable.

## 5. Flujo completo de la mutación

Para cada gen:

### Paso 1

Decidir si el gen se muta:

```text
r_i < p_m
```

### Paso 2

Calcular el rango del gen:

```text
R_i = x_max(i) - x_min(i)
```

### Paso 3

Escalar la desviación estándar:

```text
sigma_i = sigma * (R_i / 20)
```

### Paso 4

Generar ruido gaussiano:

```text
epsilon_i ~ N(0, sigma_i)
```

### Paso 5

Actualizar el gen:

```text
x'(i) = x(i) + epsilon_i
```

## 6. Ejemplo completo

Supón:

- gen actual = `50`
- `mutation_prob = 0.5`
- `sigma = 0.1`
- rango del gen = `100`

### Paso 1: decisión

Si sale:

```text
r = 0.3
```

y como:

```text
0.3 < 0.5
```

entonces el gen sí se muta.

### Paso 2: escala

```text
scaled_sigma = 0.1 * (100 / 20) = 0.5
```

### Paso 3: ruido

Supón que:

```text
epsilon = +0.4
```

### Paso 4: nuevo valor

```text
50 + 0.4 = 50.4
```

Entonces el nuevo gen queda:

```text
50.4
```

## 7. Por qué se usa una normal y no una uniforme

La distribución gaussiana tiene una propiedad útil:

- favorece cambios pequeños;
- pero mantiene la posibilidad de cambios moderados o grandes.

Eso es deseable porque en optimización continua normalmente se busca:

- hacer ajustes locales finos;
- sin perder por completo la capacidad de escapar de una región mala.

Si se usara una mutación uniforme muy amplia:

- el operador sería más brusco;
- y podría destruir con facilidad estructuras buenas ya encontradas.

## 8. Qué aporta la mutación al GA

La mutación aporta exploración local.

Mientras el cruce mezcla información entre padres, la mutación:

- introduce novedades que no estaban exactamente en ninguno de los padres;
- evita convergencia prematura;
- ayuda a explorar regiones cercanas del espacio continuo.

En este proyecto esto es especialmente importante porque:

- los genes representan parámetros continuos de membresía;
- pequeños desplazamientos pueden cambiar bastante la sensibilidad del controlador.

## 9. Relación con el código real

El operador implementado en `app/genetic/operators.py` hace exactamente esto:

```python
for i in range(len(mutated)):
    if rng.random() < mutation_prob:
        gene_range = gene_specs[i]['max'] - gene_specs[i]['min']
        scaled_sigma = sigma * (gene_range / 20.0)
        mutated[i] += rng.normal(0, scaled_sigma)
```

La interpretación del código es:

- recorrer gen por gen;
- decidir si muta;
- escalar la fuerza de la mutación según el rango del gen;
- sumar un ruido gaussiano al valor actual.

## 10. Lo que la mutación no garantiza

La mutación por sí sola no garantiza que el cromosoma resultante siga siendo válido.

Después de mutar pueden aparecer:

- valores fuera de rango;
- conjuntos desordenados;
- solapes extraños;
- huecos entre funciones de pertenencia.

Por eso el flujo correcto siempre continúa con:

- `repair()`

en `app/genetic/chromosome.py`.

La reparación:

- recorta;
- reordena;
- fuerza continuidad;
- y mantiene la interpretabilidad de las funciones difusas.

## 11. Interpretación académica

La manera correcta de explicar este operador en defensa es:

“La mutación gaussiana actúa gen por gen. Cada gen tiene cierta probabilidad de ser modificado. Si se muta, se le suma un ruido aleatorio centrado en cero. La magnitud de ese ruido se escala según el rango permitido del gen, de modo que la perturbación sea proporcional a su dominio real. Esto permite explorar soluciones cercanas sin destruir completamente las estructuras ya útiles.”

## 12. Resumen conceptual

La mutación gaussiana implementada en el proyecto cumple cuatro funciones:

- introducir variabilidad;
- mantener búsqueda local continua;
- respetar la escala propia de cada parámetro;
- y complementar al cruce BLX-α.

Su papel no es rehacer por completo un cromosoma, sino perturbarlo de forma controlada para abrir nuevas trayectorias de búsqueda.
