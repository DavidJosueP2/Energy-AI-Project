# Modelo Matemático del Entorno y la Dinámica

Este documento formaliza las ecuaciones simplificadas que usa la simulación.
La lógica difusa decide el nivel de control, pero el comportamiento temporal
del sistema se obtiene aplicando ese control sobre un modelo dinámico discreto.

## 1. Separación conceptual

El proyecto combina tres capas:

1. Entorno: genera perturbaciones exógenas como temperatura exterior, humedad,
   radiación, ocupación, tarifa y perfiles de uso.
2. Control difuso: transforma entradas crisp en un nivel de control entre
   `0` y `100`.
3. Dinámica del dispositivo: actualiza la temperatura y el consumo eléctrico
   a partir de esas perturbaciones y de la señal de control.

## 2. Temperatura exterior

La temperatura exterior se modela con un ciclo diario senoidal:

\[
T_{ext}(t) = \mu_T + A_T \sin(\phi(t)) + \varepsilon_T(t)
\]

donde:

\[
\phi(t) = \frac{2 \pi (h - h_{peak} + 6)}{24}
\]

- `mu_T`: temperatura media diaria.
- `A_T`: amplitud térmica diaria.
- `h`: hora del día.
- `h_peak`: hora del pico térmico.
- `eps_T`: ruido gaussiano suavizado.

Este perfil se implementa en `app/simulation/environment.py`.

## 3. Humedad relativa

La humedad usa el mismo ciclo, pero invertido respecto a la temperatura:

\[
H(t) = \mu_H - A_H \sin(\phi(t)) + \varepsilon_H(t)
\]

Luego se recorta al intervalo `[0.05, 0.99]`.

## 4. Radiación solar

Durante las horas de luz se usa una envolvente:

\[
R(t) = R_{max} \sin^2(\pi t_{norm}) \cdot f_{nubes}
\]

donde:

\[
t_{norm} = \frac{h - h_{sunrise}}{h_{sunset} - h_{sunrise}}
\]

El factor de nubosidad `f_nubes` introduce variabilidad sin romper la forma
diurna básica.

## 5. Tarifa eléctrica

La tarifa se define por tramos horarios y luego se normaliza para la capa
difusa:

\[
tariff_{norm} = \frac{tariff - tariff_{off}}{tariff_{on} - tariff_{off}}
\]

La tarifa normalizada entra al controlador difuso. La tarifa real en
`$/kWh` se usa para calcular el costo económico.

## 6. Carga base del hogar

El consumo eléctrico base se calcula con un factor horario:

\[
P_{base}(t) = P_{min} + (P_{max} - P_{min}) \cdot f_{hora}(t)
\]

El factor `f_hora` depende de la franja del día y se perturba levemente con
ruido para evitar patrones completamente deterministas.

## 7. Temperatura ambiente del refrigerador

El refrigerador no usa directamente la temperatura exterior. Se usa una
aproximación de temperatura interior del recinto:

\[
T_{room} = 22.5 + 0.18 (T_{ext} - 22.5) + 0.15 \cdot occupancy
\]

Luego se limita al intervalo `[18, 32]` grados Celsius.

## 8. Perturbaciones de uso del refrigerador

Se modelan dos índices:

- apertura de puerta: `door_openings`
- carga interna: `load_level`

Ambos son perfiles horarios normalizados en `[0, 1]`.

En la simulación se combinan como una sola perturbación de uso:

\[
usage\_load = door\_openings + 0.35 \cdot load\_level
\]

## 9. Entradas difusas

### HVAC

El controlador HVAC recibe:

\[
x_{hvac} = [temp\_error, humidity, occupancy, tariff_{norm}]
\]

donde:

\[
raw\_temp\_error = T_{int} - T_{obj}
\]

\[
temp\_error = |raw\_temp\_error|
\]

Además, el controlador ajusta la sensibilidad del error según la banda de
confort:

\[
temp\_error^{adj} = temp\_error \cdot \frac{comfort\_range_{ref}}{comfort\_range}
\]

Un rango de confort menor vuelve al controlador más sensible.

### Refrigerador

El controlador del refrigerador recibe:

\[
x_{refri} = [T_{refri}, door\_openings, load\_level, tariff_{norm}]
\]

## 10. Dinámica térmica general

La temperatura del dispositivo se actualiza con un balance discreto:

\[
T(k+1) = T(k)
       + \Delta_{amb}
       + \Delta_{occ}
       + \Delta_{solar}
       + \Delta_{uso}
       + \Delta_{control}
\]

donde:

\[
\Delta_{amb} = k_a (T_{amb} - T)
\]

\[
\Delta_{occ} = k_o \cdot occupancy
\]

\[
\Delta_{solar} = k_s \cdot \left(\frac{R}{1000}\right) \cdot 10
\]

\[
\Delta_{uso} = k_u \cdot usage\_load
\]

Si el nivel de control es `u \in [0,100]`, se normaliza:

\[
u_n = \frac{u}{100}
\]

y el efecto del actuador es:

\[
\Delta_{control} = k_c \cdot u_n
\]

## 11. Caso HVAC

El HVAC puede enfriar o calentar. Primero se calcula una banda muerta:

\[
deadband = \max\left(\frac{comfort\_range}{2}, 0.15\right)
\]

Si:

\[
T - T_{obj} > deadband
\]

entonces el sistema enfría:

\[
control\_effect = -\Delta_{control}
\]

Si:

\[
T - T_{obj} < -deadband
\]

entonces el sistema calienta:

\[
control\_effect = +\Delta_{control}
\]

En la banda muerta:

\[
control\_effect = 0
\]

Por tanto, para HVAC:

\[
T_{int}(k+1) = T_{int}(k)
             + \Delta_{amb}
             + \Delta_{occ}
             + \Delta_{solar}
             + control\_effect
\]

## 12. Caso refrigerador

El refrigerador solo enfría cuando el control está activo:

\[
control\_effect = -\Delta_{control}
\]

Entonces:

\[
T_{refri}(k+1) = T_{refri}(k)
               + \Delta_{amb}
               + \Delta_{uso}
               - \Delta_{control}
\]

## 13. Consumo eléctrico

Cuando el dispositivo está activo, el consumo instantáneo simplificado es:

\[
P_{device} = \frac{P_{max} \cdot u_n}{COP} + P_{standby}
\]

Si el control es prácticamente nulo, el consumo se toma como `0`.

## 14. Consumo total y costo

La simulación suma la carga base y la del dispositivo:

\[
P_{total} = P_{base} + P_{device}
\]

El costo por paso temporal es:

\[
cost_{step} = P_{total} \cdot tariff \cdot \Delta t
\]

El costo acumulado es la suma de todos los pasos.

## 15. Confort

La desviación térmica es:

\[
temp\_deviation = |T - T_{obj}|
\]

El índice de confort se define como:

\[
comfort = 1
\quad \text{si} \quad temp\_deviation \le comfort\_range
\]

y fuera de la banda decae linealmente:

\[
comfort = \max\left(0, 1 - \frac{temp\_deviation - comfort\_range}{5}\right)
\]

## 16. Interpretación

La lógica difusa no impone por sí sola la trayectoria térmica final. Lo que
hace es producir una señal de control interpretable. La forma temporal de la
temperatura surge al combinar:

- las perturbaciones del entorno,
- la física simplificada del dispositivo,
- el nivel de control difuso,
- y la banda de confort elegida.

Por eso la temperatura no sigue una línea recta exacta: el sistema busca
mantenerse dentro de una región aceptable con un compromiso entre confort,
consumo y costo.
