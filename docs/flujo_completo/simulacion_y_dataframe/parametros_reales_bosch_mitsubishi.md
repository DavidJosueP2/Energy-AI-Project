# Parámetros Reales Bosch y Mitsubishi

Este documento registra qué valores del proyecto fueron alineados con los
documentos técnicos aportados para que la simulación deje de apoyarse solo en
supuestos genéricos.

## 1. Refrigerador residencial de referencia

- Fabricante: Bosch
- Modelo: `KGN39AWCTG`
- Fuentes:
  - `Bosch Fridge Datasheet.pdf`
  - `Bosch Fridge ManualGuide.pdf`

### Datos tomados directamente de la documentación

- tipo: refrigerador-combi freestanding
- volumen total: `363 L`
- compartimento fresh food: `260 L`
- compartimento 4-star: `103 L`
- consumo anual: `162 kWh/a`
- clase climática: `Temperate, Subtropical, Tropical`
- temperatura ambiente mínima de operación: `10 C`
- temperatura ambiente máxima de operación: `43 C`
- setpoint recomendado fresh food: `4 C`
- rango configurable del compartimento refrigerador: `2 C a 8 C`
- rango configurable del congelador: `-16 C a -24 C`

### Parámetros del proyecto actualizados con esos datos

- `display_name`: `Refrigerador Bosch KGN39AWCTG`
- `default_target_temperature = 4.0`
- `target_min = 2.0`
- `target_max = 8.0`

### Parámetros que siguen siendo modelados

La documentación Bosch aportada no publica potencia instantánea nominal del
compresor ni COP. Por eso el modelo mantiene parámetros derivados:

- `max_power_kw = 0.12`
- `cop = 2.0`
- `standby_kw = 0.004`
- `control_gain = 2.50`

Estos valores no se presentan como datos del fabricante. Se usan como
calibración compatible con el consumo anual `162 kWh/a`, de forma que el
modelo tenga una demanda eléctrica media del orden correcto.

## 2. HVAC residencial de referencia

- Fabricante: Mitsubishi Electric
- Modelo interior: `MSZ-GL24NA`
- Modelo exterior: `MUZ-GL24NA`
- Fuentes:
  - `Mitsubishi HVAC DataSheet.pdf`
  - `Mitsubishi HVAC UserGuide.pdf`

### Datos tomados directamente de la documentación

- tipo: wall-mounted heat pump system
- capacidad nominal de enfriamiento: `22,400 BTU/h`
- capacidad máxima de enfriamiento: `31,400 BTU/h`
- capacidad mínima de enfriamiento: `8,200 BTU/h`
- potencia nominal de entrada en cooling: `1,800 W`
- capacidad nominal de calefacción a 47 F: `27,600 BTU/h`
- potencia nominal de entrada en heating a 47 F: `2,340 W`
- `COP` a `47 F`: `3.46`
- `SEER`: `20.5`
- `HSPF`: `10.0`
- alimentación: `208/230 V, 1 phase, 60 Hz`
- fusible recomendado: `20 A`
- rango de setpoint del control remoto: `16 C a 31 C`
- rango de operación exterior en cooling: `14 F a 115 F`
- rango de operación exterior en heating: `-4 F a 75 F`

### Conversión usada en el proyecto

La capacidad nominal de enfriamiento se convierte a kW térmicos:

\[
22{,}400\ \mathrm{BTU/h} \approx 6.56\ \mathrm{kW}
\]

### Parámetros del proyecto actualizados con esos datos

- `display_name`: `HVAC Mitsubishi MSZ-GL24NA / MUZ-GL24NA`
- `target_min = 16.0`
- `target_max = 31.0`
- `hvac_max_power_kw = 6.56`
- `hvac_cop = 3.46`

### Parámetros que siguen siendo modelados

Estos parámetros no salen directamente del datasheet y siguen siendo de
calibración del modelo térmico residencial:

- `alpha`: acoplamiento térmico de la vivienda
- `beta`: ganancia térmica por ocupación
- `gamma`: ganancia solar
- `delta = 3.45`: efecto térmico simplificado del actuador
- `initial_temperature`
- `standby_kw = 0.12`

`delta` se escaló para reflejar que el equipo Mitsubishi es más potente que el
modelo HVAC genérico anterior.

## 3. Criterio de uso académico

Con esta actualización, el proyecto distingue mejor entre:

- datos respaldados por documentación del fabricante;
- parámetros derivados por conversión de unidades;
- y parámetros de simulación que siguen siendo supuestos calibrados.

La regla de defensa recomendada es:

- si el dato sale del PDF, se presenta como especificación real del equipo;
- si el dato se infiere por conversión simple, se explica la conversión;
- si el dato pertenece al modelo dinámico, se presenta explícitamente como
  calibración del entorno residencial.
