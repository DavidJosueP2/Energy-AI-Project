# Confort, Tarifa y Trayectoria Térmica

## Propósito

Este documento explica una idea central del proyecto que suele generar dudas en la defensa: por qué la temperatura interior no sigue una línea recta exacta, cómo interviene la tarifa eléctrica en la decisión difusa y de qué manera el sistema equilibra confort térmico y costo energético.

La respuesta corta es que el controlador difuso no está diseñado para clavar un valor térmico exacto en cada instante, sino para mantener al sistema dentro de una región aceptable de operación, interpretando simultáneamente la desviación térmica, la ocupación, la humedad y la tarifa.

## Una idea equivocada frecuente

Es común pensar que un buen controlador térmico debería producir una temperatura interior prácticamente horizontal, como si el objetivo consistiera en fijar la vivienda en una línea exacta, por ejemplo `22 C` todo el tiempo.

Ese no es el criterio de este proyecto.

Aquí se usa un sistema de inferencia difusa Mamdani con variables lingüísticas. Por esa razón, el controlador no opera como un regulador rígido de error nulo permanente. Opera por regiones difusas y toma decisiones graduales. En consecuencia, la temperatura interior tiende a oscilar alrededor de una zona de confort en vez de permanecer exactamente inmóvil sobre el setpoint.

Esa oscilación no es, por sí misma, un defecto. En muchos casos es la manifestación natural de una política razonable: mantener el confort con un gasto energético moderado, evitando correcciones agresivas cuando la desviación todavía es aceptable.

## Qué significa “zona de confort”

En el proyecto, el usuario define una temperatura objetivo y un rango de confort. Si el objetivo es `22 C` y el rango es `±2 C`, entonces la banda confortable va de `20 C` a `24 C`.

Esto es importante porque el sistema no interpreta todas las desviaciones del mismo modo. Una diferencia térmica pequeña dentro de esa banda no exige la misma reacción que una diferencia grande fuera de ella. En otras palabras, el controlador distingue entre:

- una condición aún aceptable desde el punto de vista del confort;
- y una condición que ya requiere una acción más intensa.

Por eso, aunque el objetivo numérico sea un valor puntual, la lógica operativa real es regional y no puntual.

## Por qué la lógica difusa no produce una trayectoria recta

La forma de la trayectoria térmica depende de tres elementos combinados.

El primero es la representación difusa de las entradas. La temperatura no se clasifica como una sola cosa exacta, sino como una combinación de pertenencias. Un mismo valor puede pertenecer parcialmente a `confortable` y parcialmente a `alta`. Eso ya introduce una respuesta continua y no binaria.

El segundo elemento es la base de reglas. Cuando el sistema se encuentra cerca del objetivo, muchas reglas dejan de impulsar la potencia máxima y empiezan a favorecer salidas como `muy_baja`, `baja` o `media`. Eso evita sobrecorrecciones.

El tercer elemento es la dinámica física del sistema. La vivienda no cambia de temperatura instantáneamente. Hay inercia térmica, acoplamiento con el ambiente exterior, influencia de ocupación y radiación, y una capacidad finita de calefacción o enfriamiento. Por tanto, la respuesta es necesariamente gradual.

La combinación de esos tres factores hace que la curva térmica tienda a acercarse, mantenerse y volver a corregirse alrededor de una banda, en lugar de convertirse en una línea horizontal perfecta.

## El papel de la tarifa eléctrica

La tarifa eléctrica aparece como una variable lingüística de entrada del controlador difuso. Esto significa que el sistema no ignora el costo de la energía al decidir la acción de control.

Sin embargo, es importante distinguir dos planos diferentes:

La tarifa dentro del controlador es una entrada difusa. Su función es modular la agresividad de la acción. Por ejemplo, si la temperatura está solo ligeramente alejada del objetivo y la tarifa es `cara`, algunas reglas pueden preferir una acción moderada o baja en vez de una acción intensa.

El costo económico final, en cambio, no se calcula de forma difusa. Se calcula numéricamente en la simulación, multiplicando el consumo de energía por la tarifa horaria real.

Por tanto, la tarifa cumple una doble función dentro del sistema completo:

- como entrada difusa, influye en la decisión;
- como valor numérico, participa en el cálculo del costo.

## Cómo se traduce eso en el comportamiento del HVAC

En el caso del HVAC, las entradas difusas principales son:

- error térmico;
- humedad;
- ocupación;
- tarifa eléctrica.

La salida es el nivel de climatización o intensidad de control.

Si la desviación térmica es grande, la lógica difusa tiende a activar conjuntos de salida más intensos, porque la prioridad pasa a ser recuperar el confort. En cambio, si el sistema ya se encuentra dentro o muy cerca de la banda confortable, la tarifa puede influir con más fuerza en la decisión, reduciendo el nivel de control.

Eso significa que tu intuición es correcta en lo esencial: cuando la tarifa está cara y la vivienda ya se encuentra en una zona térmicamente aceptable, el sistema puede preferir no encender con tanta fuerza. No se trata de olvidar el objetivo, sino de reconocer que una corrección agresiva quizá no sea necesaria en ese instante.

En cambio, si la vivienda se aleja demasiado de la zona deseada, la lógica difusa puede volver a actuar con mayor intensidad incluso si la tarifa es alta, porque la degradación de confort se vuelve más importante.

## La interpretación correcta de la curva térmica

Desde una perspectiva académica, la curva interior no debe interpretarse como un intento fallido de seguir un valor fijo. Debe interpretarse como el resultado visible de una política de compromiso.

El controlador busca un equilibrio entre:

- cercanía al objetivo térmico;
- permanencia dentro de la banda confortable;
- moderación del consumo energético;
- sensibilidad al costo de la energía.

Por eso la curva puede subir y bajar suavemente alrededor de la zona de confort. Si el sistema ya está en una región aceptable, no siempre vale la pena gastar energía extra para forzar exactamente el centro del intervalo. La lógica difusa considera esa diferencia.

## Relación con las funciones de pertenencia

Las funciones de pertenencia hacen posible este comportamiento porque representan regiones del universo de discurso, no puntos únicos.

Por ejemplo, una región `confortable` no significa un solo valor exacto, sino un conjunto de valores alrededor de la meta con alto grado de pertenencia. Mientras la temperatura permanezca razonablemente dentro de esa región, el sistema puede mantener una acción reducida o moderada.

Cuando la temperatura se desplaza hacia regiones como `alta` o `muy_alta`, entonces aumentan las activaciones de reglas asociadas a salidas más fuertes.

Esto explica por qué la trayectoria térmica es ondulada y no rígida. La lógica difusa trabaja por transiciones suaves entre regiones semánticas, no por umbrales duros que obliguen siempre a una corrección máxima.

## Relación con el cálculo posterior del costo

Una vez que el controlador produce la salida difusa y esta se desfuzzifica, la simulación aplica esa potencia al modelo dinámico del dispositivo. Ese nivel de control determina un cierto consumo del HVAC. Luego, la simulación toma la tarifa real de esa hora y calcula el costo.

La secuencia correcta es esta:

la tarifa influye en la decisión difusa, la decisión modifica el consumo, y el consumo junto con la tarifa determina el costo final.

Por eso no debe confundirse “tarifa difusa” con “costo difuso”. La tarifa es una entrada lingüística para la toma de decisión; el costo es un resultado numérico de la simulación.

## Cómo conviene explicarlo en defensa

Una formulación sólida y defendible sería la siguiente:

“El controlador difuso no busca imponer una temperatura exactamente constante, sino mantener el sistema dentro de una región de confort térmico con un compromiso entre bienestar y costo energético. La tarifa eléctrica actúa como variable lingüística que modula la agresividad del control: cuando el sistema ya se encuentra en una zona aceptable y la tarifa es alta, la potencia se reduce; cuando la desviación térmica crece, el controlador incrementa la acción porque la prioridad vuelve a ser recuperar el confort.” 

Otra forma válida de decirlo es:

“La trayectoria interior no es una línea recta porque el sistema difuso opera por regiones de pertenencia y por reglas de compromiso. La meta no es perseguir de forma rígida un punto exacto, sino permanecer cerca de la zona deseada con una respuesta gradual y energéticamente razonable.”

## Conclusión

La forma de la temperatura interior en este proyecto no es un accidente ni una debilidad metodológica. Es la consecuencia natural de un controlador difuso interpretable que incorpora conocimiento experto, tolerancia de confort y sensibilidad al costo energético.

La tarifa eléctrica no sustituye al confort, pero sí modifica la intensidad de la respuesta cuando la situación térmica lo permite. Esa es precisamente una de las fortalezas del enfoque difuso: no imponer decisiones binarias, sino producir acciones razonables bajo criterios múltiples y parcialmente compatibles.
