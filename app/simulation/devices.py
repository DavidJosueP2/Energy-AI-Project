"""
Definiciones unificadas de dispositivos y modelo dinamico.

Este modulo es la fuente unica de verdad por dispositivo:
- identidad y etiquetas visibles;
- limites operativos y defaults de interfaz;
- dinamica fisica del modelo;
- especificacion difusa base.

La idea es que HVAC o refrigerador se definan una sola vez aqui, y que
GUI, simulacion y controlador difuso lean de esta misma definicion.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np


MembershipParams = List[float]


@dataclass(frozen=True)
class DeviceDescriptor:
    """Metadatos compartidos del dispositivo."""

    key: str
    display_name: str
    control_display_name: str
    temperature_display_name: str
    default_target_temperature: float
    default_comfort_range: float
    target_min: float
    target_max: float
    comfort_min: float
    comfort_max: float
    achievable_min_temperature: float
    achievable_max_temperature: float


@dataclass
class DeviceDynamicsConfig:
    """Configuracion fisica del modelo dinamico del dispositivo."""

    initial_temperature: float
    min_temperature: float
    max_temperature: float
    ambient_coupling: float
    occupancy_gain: float
    solar_gain: float
    usage_gain: float
    control_gain: float
    max_power_kw: float
    standby_kw: float
    cop: float


@dataclass
class VariableSpec:
    """Especifica una variable difusa de entrada o salida."""

    name: str
    display_name: str
    universe_range: Tuple[float, float]
    sets: Dict[str, MembershipParams]
    mf_types: Dict[str, str] = field(default_factory=dict)
    role: str = "input"
    optimizable: bool = True
    input_key: str = ""
    description: str = ""
    unit: str = ""

    def get_mf_type(self, set_name: str) -> str:
        return self.mf_types.get(set_name, "triangular")


@dataclass
class DeviceFuzzySpec:
    """Especificacion completa del sistema difuso del dispositivo."""

    descriptor: DeviceDescriptor
    output_name: str
    variables: List[VariableSpec]
    output_variable: VariableSpec
    explanation: str = ""

    @property
    def device_key(self) -> str:
        return self.descriptor.key

    @property
    def display_name(self) -> str:
        return self.descriptor.display_name

    @property
    def control_display_name(self) -> str:
        return self.descriptor.control_display_name

    @property
    def temperature_display_name(self) -> str:
        return self.descriptor.temperature_display_name

    @property
    def output_display_name(self) -> str:
        return self.output_variable.display_name

    @property
    def input_variables(self) -> List[VariableSpec]:
        return [variable for variable in self.variables if variable.role == "input"]

    def get_variable(self, name: str) -> VariableSpec:
        for variable in self.variables:
            if variable.name == name:
                return variable
        if self.output_variable.name == name:
            return self.output_variable
        raise KeyError(f"Variable desconocida: {name}")


@dataclass
class DeviceDefinition:
    """Definicion completa y unica de un dispositivo."""

    descriptor: DeviceDescriptor
    dynamics: DeviceDynamicsConfig
    fuzzy: DeviceFuzzySpec

    @property
    def key(self) -> str:
        return self.descriptor.key


def _build_hvac_definition() -> DeviceDefinition:
    descriptor = DeviceDescriptor(
        key="hvac",
        display_name="HVAC (Climatizacion)",
        control_display_name="Nivel de climatizacion",
        temperature_display_name="Temperatura interior",
        default_target_temperature=22.0,
        default_comfort_range=2.0,
        target_min=18.0,
        target_max=30.0,
        comfort_min=0.5,
        comfort_max=5.0,
        achievable_min_temperature=14.0,
        achievable_max_temperature=45.0,
    )
    dynamics = DeviceDynamicsConfig(
        initial_temperature=26.0,
        min_temperature=14.0,
        max_temperature=45.0,
        ambient_coupling=0.055,
        occupancy_gain=0.12,
        solar_gain=0.008,
        usage_gain=0.0,
        control_gain=1.85,
        max_power_kw=3.5,
        standby_kw=0.12,
        cop=3.2,
    )
    fuzzy_spec = DeviceFuzzySpec(
        descriptor=descriptor,
        output_name="control_output",
        explanation=(
            "Controlador difuso Mamdani para confort termico residencial. "
            "Prioriza confort, pero modula la potencia segun tarifa y ocupacion."
        ),
        variables=[
            VariableSpec(
                name="temp_error",
                display_name="Error termico",
                universe_range=(-8.0, 12.0),
                input_key="temp_error",
                description="Diferencia entre la temperatura interior y la meta termica.",
                unit="C",
                sets={
                    "baja": [-8.0, -8.0, -1.5],
                    "confortable": [-2.5, 0.0, 2.5],
                    "alta": [1.5, 4.5, 7.5],
                    "muy_alta": [6.0, 12.0, 12.0],
                },
            ),
            VariableSpec(
                name="humidity",
                display_name="Humedad",
                universe_range=(0.0, 1.0),
                input_key="humidity",
                description="Humedad relativa normalizada.",
                unit="fraccion",
                sets={
                    "baja": [0.0, 0.0, 0.35],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 1.0, 1.0],
                },
            ),
            VariableSpec(
                name="occupancy",
                display_name="Ocupacion",
                universe_range=(0.0, 6.0),
                input_key="occupancy",
                description="Numero de ocupantes presentes.",
                unit="personas",
                sets={
                    "vacia": [0.0, 0.0, 1.0],
                    "baja": [0.5, 1.5, 2.5],
                    "media": [2.0, 3.0, 4.0],
                    "alta": [3.5, 5.0, 6.0],
                },
            ),
            VariableSpec(
                name="tariff",
                display_name="Tarifa electrica",
                universe_range=(0.0, 1.0),
                input_key="tariff_normalized",
                description="Tarifa electrica normalizada.",
                unit="fraccion",
                sets={
                    "barata": [0.0, 0.0, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "cara": [0.60, 1.0, 1.0],
                },
            ),
        ],
        output_variable=VariableSpec(
            name="control_output",
            display_name=descriptor.control_display_name,
            universe_range=(0.0, 100.0),
            role="output",
            description="Potencia relativa del climatizador.",
            unit="%",
            sets={
                "muy_baja": [0.0, 0.0, 8.0],
                "baja": [5.0, 16.0, 30.0],
                "media": [24.0, 44.0, 62.0],
                "alta": [56.0, 76.0, 90.0],
                "muy_alta": [82.0, 100.0, 100.0],
            },
        ),
    )
    return DeviceDefinition(descriptor=descriptor, dynamics=dynamics, fuzzy=fuzzy_spec)


def _build_refrigerator_definition() -> DeviceDefinition:
    descriptor = DeviceDescriptor(
        key="refrigerador",
        display_name="Refrigerador",
        control_display_name="Nivel de enfriamiento",
        temperature_display_name="Temperatura interna del refrigerador",
        default_target_temperature=4.0,
        default_comfort_range=1.5,
        target_min=-2.0,
        target_max=8.0,
        comfort_min=0.5,
        comfort_max=4.0,
        achievable_min_temperature=-3.0,
        achievable_max_temperature=18.0,
    )
    dynamics = DeviceDynamicsConfig(
        initial_temperature=5.5,
        min_temperature=-3.0,
        max_temperature=18.0,
        ambient_coupling=0.060,
        occupancy_gain=0.0,
        solar_gain=0.0,
        usage_gain=0.85,
        control_gain=4.60,
        max_power_kw=0.22,
        standby_kw=0.02,
        cop=2.3,
    )
    fuzzy_spec = DeviceFuzzySpec(
        descriptor=descriptor,
        output_name="control_output",
        explanation=(
            "Controlador difuso Mamdani para refrigeracion domestica. "
            "Compensa aperturas y carga interna, moderando el consumo segun tarifa."
        ),
        variables=[
            VariableSpec(
                name="device_temperature",
                display_name="Temperatura interna",
                universe_range=(-2.0, 14.0),
                input_key="device_temperature",
                description="Temperatura interna del compartimento refrigerado.",
                unit="C",
                sets={
                    "baja": [-2.0, -2.0, 2.5],
                    "confortable": [1.5, 4.0, 6.0],
                    "alta": [5.0, 8.0, 11.0],
                    "muy_alta": [9.5, 14.0, 14.0],
                },
            ),
            VariableSpec(
                name="door_openings",
                display_name="Frecuencia de apertura",
                universe_range=(0.0, 1.0),
                input_key="door_openings",
                description="Frecuencia relativa de apertura de puerta.",
                unit="fraccion",
                sets={
                    "baja": [0.0, 0.0, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 1.0, 1.0],
                },
            ),
            VariableSpec(
                name="load_level",
                display_name="Carga interna",
                universe_range=(0.0, 1.0),
                input_key="load_level",
                description="Nivel relativo de carga termica por contenido interno.",
                unit="fraccion",
                sets={
                    "baja": [0.0, 0.0, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 1.0, 1.0],
                },
            ),
            VariableSpec(
                name="tariff",
                display_name="Tarifa electrica",
                universe_range=(0.0, 1.0),
                input_key="tariff_normalized",
                description="Tarifa electrica normalizada.",
                unit="fraccion",
                sets={
                    "barata": [0.0, 0.0, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "cara": [0.60, 1.0, 1.0],
                },
            ),
        ],
        output_variable=VariableSpec(
            name="control_output",
            display_name=descriptor.control_display_name,
            universe_range=(0.0, 100.0),
            role="output",
            description="Potencia relativa del compresor.",
            unit="%",
            sets={
                "muy_baja": [0.0, 0.0, 18.0],
                "baja": [10.0, 24.0, 38.0],
                "media": [32.0, 50.0, 66.0],
                "alta": [58.0, 76.0, 90.0],
                "muy_alta": [84.0, 100.0, 100.0],
            },
        ),
    )
    return DeviceDefinition(descriptor=descriptor, dynamics=dynamics, fuzzy=fuzzy_spec)


AVAILABLE_DEVICE_DEFINITIONS = {
    "hvac": _build_hvac_definition,
    "refrigerador": _build_refrigerator_definition,
}


def build_device_definition(device_key: str) -> DeviceDefinition:
    """Construye la definicion completa del dispositivo solicitado."""
    factory = AVAILABLE_DEVICE_DEFINITIONS.get(device_key, AVAILABLE_DEVICE_DEFINITIONS["hvac"])
    return factory()


def build_device_spec(device_key: str) -> DeviceFuzzySpec:
    """Retorna solo la especificacion difusa del dispositivo."""
    return build_device_definition(device_key).fuzzy


def build_runtime_dynamics_config(device_key: str, house_config=None) -> DeviceDynamicsConfig:
    """Construye la dinamica efectiva del dispositivo para la simulacion."""
    base_config = build_device_definition(device_key).dynamics
    if device_key != "hvac" or house_config is None:
        return base_config

    return replace(
        base_config,
        initial_temperature=house_config.initial_temperature,
        ambient_coupling=house_config.alpha,
        occupancy_gain=house_config.beta,
        solar_gain=house_config.gamma,
        control_gain=house_config.delta,
        max_power_kw=house_config.hvac_max_power_kw,
        standby_kw=house_config.hvac_standby_kw,
        cop=house_config.hvac_cop,
    )


class ControlledDevice:
    """Dispositivo termico controlado por una salida difusa en [0, 100]."""

    def __init__(
        self,
        definition: DeviceDefinition,
        target_temperature: float,
        comfort_range: float,
        dt: float = 1.0,
        dynamics_config: Optional[DeviceDynamicsConfig] = None,
    ):
        self.definition = definition
        self.descriptor = definition.descriptor
        self.dynamics = dynamics_config or definition.dynamics
        self.target_temperature = target_temperature
        self.comfort_range = comfort_range
        self.dt = dt
        self.reset()

    def reset(self, initial_temp: Optional[float] = None):
        self.temperature = initial_temp if initial_temp is not None else self.dynamics.initial_temperature
        self.power_level = 0.0
        self.consumption_kw = 0.0

    def step(
        self,
        ambient_temperature: float,
        occupancy: float,
        solar_radiation: float,
        usage_load: float = 0.0,
        control_level: Optional[float] = None,
    ) -> Dict[str, float]:
        cfg = self.dynamics
        if control_level is None:
            control_level = usage_load
            usage_load = 0.0
        ctrl_norm = float(np.clip(control_level / 100.0, 0.0, 1.0))
        self.power_level = ctrl_norm

        delta_ambient = cfg.ambient_coupling * self.dt * (ambient_temperature - self.temperature)
        delta_occupancy = cfg.occupancy_gain * self.dt * occupancy
        delta_solar = cfg.solar_gain * self.dt * (solar_radiation / 1000.0) * 10.0
        delta_usage = cfg.usage_gain * self.dt * usage_load
        delta_control = cfg.control_gain * self.dt * ctrl_norm

        self.temperature += delta_ambient + delta_occupancy + delta_solar + delta_usage - delta_control
        self.temperature = float(np.clip(self.temperature, cfg.min_temperature, cfg.max_temperature))

        if ctrl_norm > 0.01:
            self.consumption_kw = cfg.max_power_kw * ctrl_norm / max(cfg.cop, 0.1) + cfg.standby_kw
        else:
            self.consumption_kw = 0.0

        return self.get_state()

    def get_state(self) -> Dict[str, float]:
        return {
            "device_temperature": round(self.temperature, 3),
            "device_power_level": round(self.power_level, 4),
            "device_consumption_kw": round(self.consumption_kw, 4),
        }

    def get_temp_error(self) -> float:
        return self.temperature - self.target_temperature

    @property
    def display_name(self) -> str:
        return self.descriptor.display_name

    @property
    def name(self) -> str:
        return self.descriptor.key
