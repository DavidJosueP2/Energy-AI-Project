# ==============================================================================
# config.py - Configuración centralizada del sistema
# ==============================================================================
"""
Módulo de configuración global del proyecto.
Contiene todos los parámetros ajustables organizados por subsistema:
simulación, modelo térmico, controlador difuso, algoritmo genético,
métricas de evaluación y rutas de archivos.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


# ==============================================================================
# CONFIGURACIÓN DE SIMULACIÓN
# ==============================================================================
@dataclass
class SimulationConfig:
    """Parámetros de la simulación temporal."""
    
    # Horizonte temporal en horas (24h, 48h, 72h, 168h para 7 días)
    horizon_hours: int = 72
    
    # Paso de simulación en horas (0.5 = semihorario, 1.0 = horario)
    time_step_hours: float = 1.0
    
    # Semilla para reproducibilidad
    random_seed: int = 42
    
    # Temperatura objetivo de confort (°C)
    target_temperature: float = 22.0
    
    # Rango de confort aceptable alrededor del objetivo (°C)
    comfort_range: float = 2.0
    
    # Tipo de escenario: 'verano', 'invierno', 'primavera', 'mixto'
    scenario_type: str = 'verano'
    
    @property
    def num_steps(self) -> int:
        """Número total de pasos de simulación."""
        return int(self.horizon_hours / self.time_step_hours)


# ==============================================================================
# CONFIGURACIÓN DEL MODELO TÉRMICO DE LA VIVIENDA
# ==============================================================================
@dataclass
class HouseConfig:
    """
    Parámetros del modelo térmico simplificado.
    
    La ecuación de transición térmica es:
    T_int(t+1) = T_int(t) 
                 + alpha * (T_ext(t) - T_int(t))     → transferencia a través de paredes
                 + beta  * ocupacion(t)                → calor generado por personas
                 + gamma * radiacion_solar(t)          → ganancia solar
                 - delta * potencia_hvac(t)            → efecto del climatizador
    
    Donde cada coeficiente está escalado al paso temporal (dt).
    """
    
    # Coeficiente de transferencia térmica paredes (por hora)
    # Representa la conductancia térmica normalizada del edificio.
    # Valores típicos: 0.03-0.12 dependiendo del aislamiento.
    # Un valor bajo indica buen aislamiento.
    alpha: float = 0.065
    
    # Calor generado por persona (°C por persona por hora)
    # Una persona emite ~80-120W de calor. En una vivienda típica de ~80m²
    # con altura 2.5m (200m³ de aire), esto equivale a ~0.2-0.4°C/persona/hora.
    beta: float = 0.30
    
    # Coeficiente de ganancia solar (°C por unidad de radiación por hora)
    # Depende del área de ventanas, orientación y factor solar.
    gamma: float = 0.012
    
    # Efecto del climatizador (°C por unidad de potencia relativa por hora)
    # Para un sistema HVAC de ~3.5kW (12000 BTU) típico residencial.
    delta: float = 0.18
    
    # Temperatura interior inicial (°C)
    initial_temperature: float = 26.0
    
    # Potencia máxima del HVAC en kW
    hvac_max_power_kw: float = 3.5
    
    # Coeficiente de rendimiento (COP) del sistema HVAC
    # Valores típicos: 2.5-4.0 para aires acondicionados modernos.
    hvac_cop: float = 3.2
    
    # Consumo en standby del HVAC cuando está encendido pero al mínimo (kW)
    hvac_standby_kw: float = 0.15


# ==============================================================================
# CONFIGURACIÓN DEL AMBIENTE / ESCENARIOS
# ==============================================================================
@dataclass
class EnvironmentConfig:
    """Parámetros para la generación de perfiles ambientales."""
    
    # --- Temperatura exterior ---
    temp_mean: float = 30.0
    temp_amplitude: float = 7.0
    temp_peak_hour: float = 15.0
    temp_noise_std: float = 0.8
    
    # --- Humedad relativa ---
    humidity_mean: float = 0.55
    humidity_amplitude: float = 0.20
    humidity_noise_std: float = 0.05
    
    # --- Radiación solar ---
    solar_max: float = 850.0
    sunrise_hour: float = 6.5
    sunset_hour: float = 19.5
    
    # --- Ocupación ---
    max_occupants: int = 4
    
    # --- Tarifa eléctrica ($/kWh) ---
    tariff_off_peak: float = 0.08
    tariff_mid_peak: float = 0.15
    tariff_on_peak: float = 0.28
    
    # --- Consumo base ---
    base_consumption_min: float = 0.4
    base_consumption_max: float = 1.8


# ==============================================================================
# CONFIGURACIÓN DEL CONTROLADOR DIFUSO
# ==============================================================================
@dataclass
class FuzzyConfig:
    """
    Parámetros del controlador difuso.
    Define los universos de discurso y los parámetros iniciales
    de las funciones de pertenencia.
    """
    
    # Resolución del universo de discurso (puntos de evaluación)
    universe_resolution: int = 200
    
    # --- Universos de discurso ---
    temp_error_range: Tuple[float, float] = (-10.0, 15.0)
    humidity_range: Tuple[float, float] = (0.0, 1.0)
    occupancy_range: Tuple[float, float] = (0.0, 6.0)
    tariff_range: Tuple[float, float] = (0.0, 1.0)
    consumption_range: Tuple[float, float] = (0.0, 1.0)
    output_range: Tuple[float, float] = (0.0, 100.0)
    
    # --- Parámetros de funciones de pertenencia (triangulares: [a, b, c]) ---
    # Error de temperatura
    temp_error_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'muy_frio':      [-10.0, -10.0, -4.0],
        'frio':          [-6.0,  -3.0,   0.0],
        'confortable':   [-2.0,   0.0,   2.0],
        'calido':        [ 1.0,   3.5,   6.0],
        'caliente':      [ 4.0,   6.5,   9.0],
        'muy_caliente':  [ 7.0,  11.0,  15.0],
    })
    
    # Humedad relativa normalizada [0, 1]
    humidity_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'baja':  [0.0, 0.0,  0.35],
        'media': [0.25, 0.50, 0.75],
        'alta':  [0.65, 1.0,  1.0],
    })
    
    # Ocupacion
    occupancy_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'vacia': [0.0, 0.0, 1.0],
        'baja':  [0.5, 1.5, 2.5],
        'media': [2.0, 3.0, 4.0],
        'alta':  [3.0, 5.0, 6.0],
    })
    
    # Tarifa electrica normalizada
    tariff_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'barata':  [0.0, 0.0,  0.35],
        'media':   [0.2, 0.45, 0.70],
        'cara':    [0.55, 0.85, 1.0],
    })
    
    # Consumo actual normalizado
    consumption_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'bajo':  [0.0, 0.0,  0.35],
        'medio': [0.2, 0.45, 0.70],
        'alto':  [0.55, 0.85, 1.0],
    })
    
    # Salida: nivel de climatización [0, 100]
    output_sets: Dict[str, List[float]] = field(default_factory=lambda: {
        'muy_baja': [ 0.0,   0.0,  20.0],
        'baja':     [10.0,  25.0,  40.0],
        'media':    [30.0,  50.0,  65.0],
        'alta':     [55.0,  75.0,  85.0],
        'muy_alta': [75.0, 100.0, 100.0],
    })


# ==============================================================================
# CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ==============================================================================
@dataclass
class GeneticConfig:
    """Parámetros del algoritmo genético para optimización del controlador."""
    
    # Tamaño de la población
    population_size: int = 60
    
    # Número de generaciones
    num_generations: int = 40
    
    # Probabilidad de cruce
    crossover_probability: float = 0.85
    
    # Probabilidad de mutación por gen
    mutation_probability: float = 0.12
    
    # Desviación estándar de la mutación gaussiana
    mutation_sigma: float = 0.8
    
    # Tamaño del torneo de selección
    tournament_size: int = 3
    
    # Número de individuos élite preservados
    elitism_count: int = 2
    
    # Factor alfa para cruce BLX-α
    blx_alpha: float = 0.5
    
    # Semilla del GA (puede diferir de la simulación)
    random_seed: int = 42
    
    # Margen mínimo entre puntos de funciones de pertenencia adyacentes
    min_mf_separation: float = 0.5
    
    # Rango de perturbación máxima para inicialización
    init_perturbation: float = 2.0


# ==============================================================================
# CONFIGURACIÓN DE MÉTRICAS Y FITNESS
# ==============================================================================
@dataclass
class MetricsConfig:
    """Pesos y parámetros para la función de fitness multiobjetivo."""
    
    # Peso para el score de confort (mayor = más importancia al confort)
    weight_comfort: float = 0.40
    
    # Peso para el ahorro energético
    weight_energy: float = 0.25
    
    # Peso para el ahorro de costo
    weight_cost: float = 0.20
    
    # Peso (penalización) para el pico de demanda
    weight_peak: float = 0.10
    
    # Peso (penalización) para la variabilidad del control
    weight_variability: float = 0.05
    
    # Temperatura confort mínima y máxima para cálculo de confort
    comfort_min: float = 20.0
    comfort_max: float = 24.0


# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
@dataclass
class PathsConfig:
    """Rutas de archivos y directorios del proyecto."""
    
    # Directorio raíz del proyecto
    project_root: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))
    
    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, 'data')
    
    @property
    def scenarios_dir(self) -> str:
        return os.path.join(self.data_dir, 'scenarios')
    
    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.data_dir, 'outputs')
    
    def ensure_dirs(self):
        """Crea los directorios necesarios si no existen."""
        for d in [self.data_dir, self.scenarios_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)


# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
@dataclass
class AppConfig:
    """Configuración global de la aplicación que agrupa todos los subsistemas."""
    
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    house: HouseConfig = field(default_factory=HouseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def __post_init__(self):
        """Inicialización post-creación: asegura directorios."""
        self.paths.ensure_dirs()
        # Sincronizar confort entre simulation y metrics
        self.metrics.comfort_min = self.simulation.target_temperature - self.simulation.comfort_range
        self.metrics.comfort_max = self.simulation.target_temperature + self.simulation.comfort_range


def get_default_config() -> AppConfig:
    """Retorna la configuración por defecto de la aplicación."""
    return AppConfig()
