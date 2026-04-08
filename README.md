# 🏠 Sistema Inteligente de Gestión Energética Residencial

## Proyecto de Inteligencia Artificial — Control Difuso + Optimización Genética

---

## 📋 Descripción General

Este proyecto implementa una **simulación completa de una vivienda inteligente** que gestiona el consumo eléctrico y el confort térmico a lo largo del tiempo, utilizando dos técnicas fundamentales de Inteligencia Artificial:

1. **Lógica Difusa (Fuzzy Logic):** Un controlador tipo Mamdani que decide el nivel de climatización basándose en múltiples variables del entorno (temperatura, ocupación, tarifa eléctrica, consumo actual). El controlador captura conocimiento experto mediante reglas lingüísticas que balancean confort y eficiencia.

2. **Algoritmo Genético (GA):** Un optimizador evolutivo que ajusta automáticamente los parámetros del controlador difuso (puntos de las funciones de pertenencia) para encontrar la configuración que maximice una función de fitness multiobjetivo.

El sistema incluye una **interfaz gráfica profesional** con PyQt5 que permite ejecutar simulaciones, visualizar resultados en tiempo real, optimizar el controlador y comparar rendimiento antes/después de la optimización.

---

## 🎯 Objetivos del Proyecto

- Demostrar la aplicación práctica de lógica difusa en control inteligente
- Implementar un algoritmo genético para optimización de parámetros
- Simular dinámicas térmicas residenciales realistas
- Combinar múltiples objetivos (confort, energía, costo) en una métrica integrada
- Visualizar resultados de forma profesional y defendible académicamente

---

## 🧠 Fundamento Conceptual

### Modelo Térmico de la Vivienda

La vivienda se modela con una ecuación de balance térmico discretizada:

```
T_int(t+1) = T_int(t) + α·Δt·(T_ext - T_int) + β·Δt·N_occ + γ·Δt·R_solar - δ·Δt·P_hvac
```

| Coeficiente | Significado | Valor | Justificación |
|---|---|---|---|
| α | Conductancia térmica | 0.065 /h | Aislamiento típico residencial |
| β | Calor por ocupante | 0.30 °C/persona/h | ~100W por persona en ~200m³ |
| γ | Ganancia solar | 0.012 | Ventanas estándar con factor solar |
| δ | Efecto HVAC | 0.18 °C/h | Split 3.5kW, COP 3.2 |

### Controlador Difuso (Mamdani)

**Variables de entrada:**
- **Error de temperatura:** T_interior - T_objetivo → {muy_frío, frío, confortable, cálido, caliente, muy_caliente}
- **Ocupación:** personas presentes → {vacía, baja, media, alta}
- **Tarifa eléctrica:** normalizada [0,1] → {barata, media, cara}
- **Consumo actual:** normalizado [0,1] → {bajo, medio, alto}

**Variable de salida:**
- **Nivel HVAC:** [0, 100] → {muy_baja, baja, media, alta, muy_alta}

**Motor de inferencia:**
- Fuzzificación con funciones triangulares parametrizables
- Evaluación de reglas con operador AND (mínimo)
- Implicación por recorte (mínimo)
- Agregación por máximo
- Desfuzzificación por centroide

**Base de reglas:** 35 reglas diseñadas con los siguientes principios:
- Priorizar confort cuando hay ocupación alta
- Moderar climatización con tarifa cara
- Evitar picos de consumo
- Minimizar operación innecesaria en zona de confort

### Algoritmo Genético

**Cromosoma:** Vector de valores reales codificando los parámetros [a, b, c] de cada función de pertenencia triangular de todas las variables.

**Operadores:**
| Operador | Implementación | Parámetros |
|---|---|---|
| Selección | Torneo | Tamaño 3 |
| Cruce | BLX-α | α = 0.5 |
| Mutación | Gaussiana | σ adaptativa |
| Elitismo | Preservación | Top 2 |
| Reparación | Orden a≤b≤c | Automática |

**Función de Fitness:**
```
fitness = w₁·confort + w₂·(1-energía) + w₃·(1-costo) - w₄·pico - w₅·variabilidad
```

| Peso | Objetivo | Valor |
|---|---|---|
| w₁ | Confort térmico | 0.35 |
| w₂ | Ahorro energético | 0.25 |
| w₃ | Ahorro de costo | 0.20 |
| w₄ | Penalización pico | 0.10 |
| w₅ | Suavidad de control | 0.10 |

---

## 📁 Arquitectura del Proyecto

```
energy_ai_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py              ← Punto de entrada (GUI/CLI)
│   ├── config.py             ← Configuración centralizada
│   │
│   ├── simulation/           ← Modelo y motor de simulación
│   │   ├── environment.py    ← Perfiles ambientales (T°, solar, ocupación, tarifa)
│   │   ├── house_model.py    ← Modelo térmico de la vivienda
│   │   ├── simulator.py      ← Motor de simulación temporal
│   │   ├── scenario_generator.py ← Escenarios predefinidos
│   │   └── metrics.py        ← Métricas de desempeño y fitness
│   │
│   ├── fuzzy/                ← Sistema de lógica difusa
│   │   ├── membership.py     ← Funciones de pertenencia (triangular, trapezoidal)
│   │   ├── rules.py          ← Base de 35 reglas difusas
│   │   ├── inference.py      ← Motor de inferencia Mamdani
│   │   └── controller.py     ← Controlador difuso de alto nivel
│   │
│   ├── genetic/              ← Algoritmo genético
│   │   ├── chromosome.py     ← Codificación/decodificación de cromosomas
│   │   ├── fitness.py        ← Evaluación de fitness
│   │   ├── operators.py      ← Selección, cruce, mutación
│   │   ├── optimizer.py      ← Bucle principal del GA
│   │   └── evaluation.py     ← Evaluación comparativa
│   │
│   ├── visualization/        ← Gráficos y reportes
│   │   ├── plots.py          ← 10+ gráficos con matplotlib (tema oscuro)
│   │   ├── dashboard.py      ← Dashboards multiplot
│   │   └── report_export.py  ← Exportación CSV, PNG, HTML
│   │
│   └── ui/                   ← Interfaz gráfica
│       └── gui.py            ← GUI completa con PyQt5
│
├── data/
│   ├── scenarios/            ← Datos de escenarios
│   └── outputs/              ← Resultados exportados
│
├── tests/
│   ├── test_simulation.py    ← Tests de simulación
│   ├── test_fuzzy.py         ← Tests de lógica difusa
│   └── test_genetic.py       ← Tests del GA
│
├── requirements.txt
├── README.md
└── run.py                    ← Script de lanzamiento
```

---

## 🚀 Instalación

### Requisitos previos
- Python 3.9 o superior
- pip (gestor de paquetes)

### Pasos

1. **Clonar o copiar el proyecto:**
```bash
cd energy_ai_project
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

### Modo GUI (recomendado)
```bash
python run.py
```

Esto abre la interfaz gráfica profesional donde puede:
1. Configurar parámetros de simulación (duración, escenario, temperatura objetivo)
2. Ejecutar simulación base → ver gráficos y métricas
3. Ejecutar optimización genética → ver evolución del fitness
4. Comparar resultados base vs optimizado
5. Exportar reporte HTML o CSV

### Modo CLI (línea de comandos)
```bash
# Simulación base con reporte
python run.py --cli --scenario verano --hours 72

# Con optimización genética
python run.py --cli --optimize --scenario verano --hours 48

# Exportar a ruta específica
python run.py --cli --optimize --export ./mi_reporte.html
```

### Ejecutar pruebas
```bash
pytest tests/ -v
```

---

## 📊 Gráficos Incluidos

| # | Gráfico | Descripción |
|---|---|---|
| 1 | Temperaturas | Interior vs exterior con zona de confort |
| 2 | HVAC | Señal de control del climatizador |
| 3 | Consumo | Desglose base + HVAC por hora |
| 4 | Costo | Costo eléctrico acumulado |
| 5 | Confort | Desviación térmica e índice de confort |
| 6 | GA Evolution | Mejor y promedio fitness por generación |
| 7 | Comparación | Base vs optimizado (temp, HVAC, costo) |
| 8 | Distribución | Histograma y boxplot de consumo |
| 9 | Ocupación/Tarifa | Perfiles temporales |
| 10 | Métricas | Barras comparativas de indicadores clave |

---

## 📈 Métricas de Evaluación

El sistema calcula y compara:

- **Energía total** (kWh): consumo acumulado durante la simulación
- **Costo total** ($): gasto eléctrico según tarifa variable
- **Confort** (%): porcentaje del tiempo dentro del rango objetivo
- **Pico de demanda** (kW): máxima demanda instantánea
- **Nivel HVAC promedio** (%): intensidad media del climatizador
- **Variabilidad de control**: suavidad de la señal HVAC
- **Fitness score**: métrica compuesta multiobjetivo

---

## 🔧 Escenarios Disponibles

| Escenario | T. Media | Amplitud | Radiación | Descripción |
|---|---|---|---|---|
| Verano | 32°C | 8°C | 950 W/m² | Calor intenso, alta demanda |
| Invierno | 12°C | 5°C | 450 W/m² | Frío, baja radiación |
| Primavera | 24°C | 6°C | 700 W/m² | Clima templado y benigno |
| Mixto | 27°C | 9°C | 750 W/m² | Condiciones variables |

---

## 🧪 Datos Simulados

Todos los datos son generados algorítmicamente con semilla fija para reproducibilidad:

- **Temperatura exterior:** Sinusoidal diaria + ruido gaussiano suavizado
- **Radiación solar:** Campana sin² durante horas diurnas + variabilidad por nubes
- **Ocupación:** Patrón residencial típico (dormir, trabajo, cena)
- **Tarifa:** Franjas horarias (valle 0-8h, llano 8-10h/14-18h/22-24h, punta 10-14h/18-22h)
- **Consumo base:** Perfil con picos en mañana y noche (cocina, iluminación)

No se requiere conexión a internet ni APIs externas.

---

## 🎓 Uso Académico

Este proyecto está diseñado para ser presentado como proyecto universitario de Inteligencia Artificial. Incluye:

- ✅ Fundamentación teórica documentada en código y README
- ✅ Arquitectura modular y profesional
- ✅ Implementación propia de lógica difusa (no librería black-box)
- ✅ Implementación propia del algoritmo genético
- ✅ Comparación cuantitativa base vs optimizado
- ✅ Visualización profesional con tema oscuro
- ✅ Pruebas automatizadas
- ✅ Interfaz gráfica funcional para demostración en vivo
- ✅ Exportación de reportes para documentación

---

## ⚖️ Licencia

Proyecto académico de uso educativo.

---

*Desarrollado como proyecto de Inteligencia Artificial — Gestión Energética Residencial con Lógica Difusa y Algoritmo Genético.*
