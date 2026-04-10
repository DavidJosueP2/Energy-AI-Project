# Cruce BLX-α

## Objetivo

Este documento explica en detalle el operador de cruce `BLX-alpha` usado en el proyecto.

Su propósito es mostrar:

- qué significa el rango de los padres;
- cómo se expande ese rango;
- cómo se generan los hijos;
- y por qué este operador es adecuado para cromosomas reales.

## Contexto dentro del proyecto

En este sistema, un cromosoma está formado por genes reales que codifican parámetros de funciones de pertenencia.

Por tanto, el cruce no combina bits ni etiquetas discretas. Combina valores continuos.

Eso hace que un operador como `BLX-alpha` sea apropiado, porque:

- trabaja gen por gen;
- genera hijos continuos;
- y permite explorar regiones cercanas a los padres sin quedarse solo en promedios.

El código real está en:

- `app/genetic/operators.py`

## Idea intuitiva

BLX-α puede entenderse así:

1. tomar dos padres;
2. observar, para cada gen, el intervalo definido por sus valores;
3. ampliar ese intervalo hacia ambos lados;
4. generar valores hijos dentro de ese intervalo ampliado.

La intuición central es:

- si dos padres tienen genes parecidos, probablemente esa región del espacio es prometedora;
- pero no conviene explorar solo exactamente entre ellos;
- conviene permitir una expansión moderada que habilite soluciones nuevas cercanas.

## 1. Qué significa “rango de los padres”

Para un gen `i`, cada padre tiene un valor:

- `p1(i)`
- `p2(i)`

Por ejemplo:

- `p1(i) = 18`
- `p2(i) = 22`

Entonces se define:

```text
p_min(i) = min(p1(i), p2(i))
p_max(i) = max(p1(i), p2(i))
```

En el ejemplo:

```text
p_min(i) = 18
p_max(i) = 22
```

El intervalo:

```text
[18, 22]
```

es el rango natural de los padres para ese gen.

## 2. Qué representa ese rango

Ese intervalo representa la región del espacio de búsqueda que ya está respaldada por ambos padres.

Si el hijo se generara únicamente dentro de ese intervalo:

- el algoritmo estaría recombinando información ya conocida;
- pero no estaría explorando más allá de lo que los padres ya contienen.

## 3. Qué significa “expandir el rango”

BLX-α no se queda solo en el intervalo original.

Primero calcula la distancia entre padres:

```text
d(i) = p_max(i) - p_min(i)
```

Si `p1(i)=18` y `p2(i)=22`, entonces:

```text
d(i) = 22 - 18 = 4
```

Después usa el parámetro `alpha` para extender el intervalo:

```text
L(i) = p_min(i) - alpha * d(i)
U(i) = p_max(i) + alpha * d(i)
```

Si `alpha = 0.5`, entonces:

```text
L(i) = 18 - 0.5 * 4 = 16
U(i) = 22 + 0.5 * 4 = 24
```

El intervalo original:

```text
[18, 22]
```

se convierte en:

```text
[16, 24]
```

Eso es exactamente lo que significa expandir el rango.

## 4. Por qué se expande

La expansión se usa para evitar que el algoritmo se vuelva demasiado conservador.

Si los hijos solo pudieran quedar entre los padres:

- el algoritmo sería muy explotativo;
- combinaría soluciones conocidas;
- pero le costaría salir de una región estrecha del espacio.

La expansión permite:

- probar valores un poco menores que ambos padres;
- probar valores un poco mayores que ambos padres;
- explorar soluciones cercanas no vistas todavía.

En términos de búsqueda:

- sin expansión, el algoritmo solo mezcla lo que ya tiene;
- con expansión, el algoritmo mezcla y además explora.

## 5. Fórmula completa de BLX-α

Para cada gen `i`:

### Paso 1: identificar extremos

```text
p_min(i) = min(p1(i), p2(i))
p_max(i) = max(p1(i), p2(i))
```

### Paso 2: calcular distancia

```text
d(i) = p_max(i) - p_min(i)
```

### Paso 3: expandir intervalo

```text
L(i) = p_min(i) - alpha * d(i)
U(i) = p_max(i) + alpha * d(i)
```

donde:

- `L(i)` es el límite inferior;
- `U(i)` es el límite superior.

### Paso 4: generar hijos

Cada hijo se genera muestreando uniformemente en ese intervalo:

```text
h1(i) ~ U(L(i), U(i))
h2(i) ~ U(L(i), U(i))
```

## 6. Qué significa “uniforme”

Uniforme significa que cualquier valor dentro del intervalo tiene la misma probabilidad relativa de ser elegido.

Si el intervalo es:

```text
[16, 24]
```

pueden salir valores como:

- `16.3`
- `17.8`
- `19.5`
- `21.2`
- `23.7`

Todos son válidos y todos pertenecen a la misma distribución uniforme sobre el intervalo.

Esto implica:

- no se toma un promedio fijo;
- no se copia directamente un valor parental;
- no se favorece el centro por defecto.

## 7. Ejemplo completo con un gen

Supón:

- `p1(i) = 10`
- `p2(i) = 20`
- `alpha = 0.5`

### Paso 1

```text
p_min(i) = 10
p_max(i) = 20
```

### Paso 2

```text
d(i) = 20 - 10 = 10
```

### Paso 3

```text
L(i) = 10 - 0.5 * 10 = 5
U(i) = 20 + 0.5 * 10 = 25
```

### Paso 4

```text
h1(i) ~ U(5, 25)
h2(i) ~ U(5, 25)
```

Entonces podrían salir:

- `h1(i) = 8.4`
- `h2(i) = 22.1`

o también:

- `h1(i) = 14.7`
- `h2(i) = 24.3`

La observación importante es:

- un hijo puede quedar fuera del intervalo original `[10,20]`;
- eso no es un error;
- es parte intencional del mecanismo de exploración.

## 8. Ejemplo con varios genes

Supón dos padres:

```text
p1 = (10, 30, 0.4)
p2 = (20, 50, 0.8)
```

BLX-α trabaja gen por gen:

- para el gen 1 construye un intervalo expandido;
- para el gen 2 construye otro;
- para el gen 3 construye otro.

Después muestrea cada gen por separado y arma el cromosoma hijo completo:

```text
h1 = (h1(1), h1(2), h1(3))
h2 = (h2(1), h2(2), h2(3))
```

Esto es importante porque muestra que el operador es vectorial solo en apariencia:

- en realidad trata cada dimensión del cromosoma de forma independiente.

## 9. Relación con el código del proyecto

El código implementa exactamente esta idea:

```python
p_min = min(parent1[i], parent2[i])
p_max = max(parent1[i], parent2[i])
d = p_max - p_min

low = p_min - alpha * d
high = p_max + alpha * d

child1[i] = rng.uniform(low, high)
child2[i] = rng.uniform(low, high)
```

Eso aparece en:

- `blx_alpha_crossover(...)` en `app/genetic/operators.py`

La interpretación matemática del código es exactamente:

```text
p_min(i) = min(p1(i), p2(i))
p_max(i) = max(p1(i), p2(i))
d(i) = p_max(i) - p_min(i)
L(i) = p_min(i) - alpha * d(i)
U(i) = p_max(i) + alpha * d(i)
h1(i), h2(i) ~ U(L(i), U(i))
```

## 10. Qué pasa si `alpha = 0`

Si `alpha = 0`, no hay expansión.

Entonces:

```text
L(i) = p_min(i)
U(i) = p_max(i)
```

Los hijos solo pueden quedar entre los padres.

Esto produce un comportamiento más conservador.

## 11. Qué pasa si `alpha > 0`

Si `alpha > 0`, sí hay expansión.

Mientras mayor sea `alpha`:

- mayor es la región de exploración;
- mayor es la diversidad potencial de los hijos;
- mayor es el riesgo de salir de regiones buenas conocidas.

Por eso `alpha` controla el equilibrio entre:

- explotación de soluciones ya prometedoras;
- y exploración de nuevas variantes.

## 12. Por qué BLX-α es adecuado aquí

En este proyecto los genes representan parámetros reales de funciones de pertenencia difusas.

Por tanto:

- no se necesita un cruce binario;
- no tiene sentido hacer recombinación simbólica de reglas;
- sí conviene un operador continuo.

BLX-α es adecuado porque:

- respeta la naturaleza real de los genes;
- combina información de ambos padres;
- permite explorar alrededor de soluciones buenas;
- y es simple de explicar académicamente.

## 13. Limitación importante

BLX-α por sí solo no garantiza que el hijo represente una familia de funciones de pertenencia válida.

Por eso, después del cruce:

- se aplica mutación;
- y luego una reparación estructural en `ChromosomeEncoder.repair(...)`.

Sin esa reparación, podrían aparecer:

- conjuntos desordenados;
- solapes incoherentes;
- huecos sin cobertura;
- o extremos que no alcancen el universo.

## 14. Forma breve para exponer

Una forma correcta y defendible de explicarlo es:

“En BLX-α, para cada gen se identifica el valor mínimo y máximo entre los dos padres. Ese intervalo se expande en ambos extremos usando el parámetro α, y luego los hijos se generan aleatoriamente con distribución uniforme dentro de ese rango ampliado. De ese modo, el operador no solo combina soluciones existentes, sino que también explora nuevas variantes cercanas.”
