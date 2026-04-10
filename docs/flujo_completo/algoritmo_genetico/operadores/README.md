# Operadores Genéticos

## Propósito de esta carpeta

Esta carpeta documenta con detalle formal los operadores genéticos usados en el proyecto.

El objetivo es dejar claro:

- qué hace cada operador;
- por qué se eligió;
- cómo se relaciona con el código real;
- y cómo debe explicarse en una defensa académica.

## Operadores implementados en el proyecto

Los operadores reales están en:

- `app/genetic/operators.py`

Los operadores usados por el algoritmo genético son:

- selección por torneo;
- cruce `BLX-alpha`;
- mutación gaussiana;
- elitismo.

## Documentos específicos

- [Cruce BLX-α](./cruce_blx_alpha.md)
- [Mutación Gaussiana](./mutacion_gaussiana.md)
- [Elitismo](./elitismo.md)

## Relación con el flujo general del GA

Dentro del ciclo de evolución en `app/genetic/optimizer.py`, el orden operacional es:

1. seleccionar padres;
2. generar hijos por cruce;
3. mutar genes de los hijos;
4. reparar restricciones del cromosoma;
5. evaluar la población resultante;
6. aplicar elitismo.

Esto significa que los operadores no actúan aisladamente. Forman una secuencia con propósito:

- la selección define qué soluciones se reproducen;
- el cruce combina información de soluciones prometedoras;
- la mutación introduce nuevas variantes locales;
- la reparación garantiza que las funciones de pertenencia sigan siendo válidas;
- el elitismo evita perder soluciones buenas ya encontradas.

## Importancia académica

En este proyecto, los operadores genéticos no se aplican sobre decisiones directas de control ni sobre reglas simbólicas arbitrarias.

Se aplican sobre cromosomas reales que codifican parámetros de funciones de pertenencia difusas.

Por eso su interpretación correcta es:

- el cruce mezcla configuraciones paramétricas del sistema difuso;
- la mutación perturba esas configuraciones de forma controlada;
- y la reparación mantiene la coherencia semántica de las particiones difusas.

## Forma breve de explicarlo

Una forma correcta y defendible de describir esta carpeta es:

“Los operadores genéticos trabajan sobre los parámetros de las funciones de pertenencia del controlador difuso. La selección elige soluciones prometedoras, el cruce BLX-α combina sus genes en un espacio continuo expandido, la mutación gaussiana introduce variaciones locales proporcionales a la escala de cada gen y la etapa de reparación asegura que las funciones resultantes sigan siendo válidas e interpretables.”
