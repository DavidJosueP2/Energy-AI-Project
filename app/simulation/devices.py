from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np


MembershipParams = List[float]


@dataclass(frozen=True)
class DeviceDescriptor:
    """Metadatos compartidos del dispositivo."""

    key: str
    manufacturer: str
    model: str
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
    """Configuracion fisica del modelo dinamico del dispositivo.

    Los coeficientes de esta estructura gobiernan el balance termico discreto
    aplicado en ``ControlledDevice.step(...)``.
    """

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
        manufacturer="Mitsubishi Electric",
        model="MSZ-GL24NA / MUZ-GL24NA",
        display_name="HVAC Mitsubishi MSZ-GL24NA / MUZ-GL24NA",
        control_display_name="Nivel de climatizacion Mitsubishi",
        temperature_display_name="Temperatura interior controlada",
        # Banda HVAC inicial aproximada tomada de la zona ASHRAE de 0.1 m/s:
        # 22.0 °C a 25.5 °C -> centro 23.75 °C con semi-ancho 1.75 °C.
        default_target_temperature=23.75,
        default_comfort_range=1.75,
        target_min=16.0,
        target_max=31.0,
        comfort_min=0.5,
        comfort_max=5.0,
        achievable_min_temperature=14.0,
        achievable_max_temperature=45.0,
    )
    # Datos respaldados por Mitsubishi:
    # - rango de consigna del control remoto: 16 a 31 C
    # - capacidad nominal de enfriamiento: 22,400 BTU/h = 6.56 kW termicos
    # - COP a 47 F: 3.46
    # - potencia electrica nominal en cooling: 1,800 W
    # - Temperatura de operacion: ashrae (20 a 24 °C)
    #
    # Supuestos que permanecen como calibracion del modelo:
    # - control_gain: se escala respecto al modelo anterior para reflejar la
    #   mayor capacidad termica del equipo
    # - standby_kw: se mantiene como sobreconsumo simplificado del sistema
    dynamics = DeviceDynamicsConfig(
        initial_temperature=26.0,
        min_temperature=14.0,
        max_temperature=45.0,
        ambient_coupling=0.055,
        occupancy_gain=0.12,
        solar_gain=0.008,
        usage_gain=0.0,
        control_gain=3.45,
        max_power_kw=6.56,
        standby_kw=0.12,
        cop=3.46,
    )
    fuzzy_spec = DeviceFuzzySpec(
        descriptor=descriptor,
        output_name="control_output",
        explanation=(
            "Controlador difuso Mamdani para confort termico residencial "
            "basado en un equipo Mitsubishi MSZ-GL24NA / MUZ-GL24NA. "
            "Prioriza confort, pero modula la potencia segun tarifa y ocupacion."
        ),
        variables=[
            VariableSpec(
                name="temp_error",
                display_name="Error termico",
                universe_range=(-8.0, 8.0),
                input_key="temp_error",
                description=(
                    "Diferencia con signo entre la temperatura interior y la meta termica. "
                    "Valores negativos indican ambiente mas frio que el objetivo; "
                    "valores positivos indican ambiente mas caliente."
                ),
                unit="C",
                sets={
                    # Base tecnica:
                    # - Mitsubishi distingue cercania al setpoint entre 1 y 2 C.
                    # - A partir de ~2 C el equipo sigue operando para alcanzar la consigna.
                    # - ASHRAE respalda una zona de confort como banda, no como punto.
                    "baja": [-8.0, -8.0, -6.0, -1.5],
                    "confortable": [-1.5, 0.0, 1.5],
                    "alta": [1.0, 2.0, 4.0],
                    "muy_alta": [3.0, 5.5, 8.0, 8.0],
                },
                mf_types={
                    "baja": "trapezoidal",
                    "muy_alta": "trapezoidal",
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
                    "baja": [0.0, 0.0, 0.15, 0.35],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 0.85, 1.0, 1.0],
                },
                mf_types={
                    "baja": "trapezoidal",
                    "alta": "trapezoidal",
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
                    "vacia": [0.0, 0.0, 0.4, 1.0],
                    "baja": [0.5, 1.5, 2.5],
                    "media": [2.0, 3.0, 4.0],
                    "alta": [3.5, 5.0, 6.0, 6.0],
                },
                mf_types={
                    "vacia": "trapezoidal",
                    "alta": "trapezoidal",
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
                    "barata": [0.0, 0.0, 0.12, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "cara": [0.60, 0.85, 1.0, 1.0],
                },
                mf_types={
                    "barata": "trapezoidal",
                    "cara": "trapezoidal",
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
                "muy_baja": [0.0, 0.0, 3.0, 8.0],
                "baja": [5.0, 16.0, 30.0],
                "media": [24.0, 44.0, 62.0],
                "alta": [56.0, 76.0, 90.0],
                "muy_alta": [82.0, 95.0, 100.0, 100.0],
            },
            mf_types={
                "muy_baja": "trapezoidal",
                "muy_alta": "trapezoidal",
            },
        ),
    )
    return DeviceDefinition(descriptor=descriptor, dynamics=dynamics, fuzzy=fuzzy_spec)


def _build_refrigerator_definition() -> DeviceDefinition:
    descriptor = DeviceDescriptor(
        key="refrigerador",
        manufacturer="Bosch",
        model="KGN39AWCTG",
        display_name="Refrigerador Bosch KGN39AWCTG",
        control_display_name="Nivel de enfriamiento Bosch",
        temperature_display_name="Temperatura interna del compartimento fresh food",
        default_target_temperature=4.0,
        default_comfort_range=1.5,
        target_min=2.0,
        target_max=8.0,
        comfort_min=0.5,
        comfort_max=4.0,
        achievable_min_temperature=0.0,
        achievable_max_temperature=18.0,
    )
    # Datos respaldados por Bosch:
    # - setpoint recomendado fresh food: 4 C
    # - rango configurable del compartimento de refrigeracion: 2 a 8 C
    # - consumo anual: 162 kWh/a
    # - clima de operacion: 10 a 43 C
    #
    # Datos derivados para el modelo dinamico:
    # - max_power_kw y standby_kw se calibran para ser compatibles con el
    #   consumo anual publicado, ya que la ficha no reporta potencia nominal
    #   instantanea del compresor.
    dynamics = DeviceDynamicsConfig(
        initial_temperature=5.5,
        min_temperature=0.0,
        max_temperature=18.0,
        ambient_coupling=0.060,
        occupancy_gain=0.0,
        solar_gain=0.0,
        usage_gain=0.85,
        control_gain=2.50,
        max_power_kw=0.12,
        standby_kw=0.004,
        cop=2.0,
    )
    fuzzy_spec = DeviceFuzzySpec(
        descriptor=descriptor,
        output_name="control_output",
        explanation=(
            "Controlador difuso Mamdani para refrigeracion domestica "
            "basado en un Bosch KGN39AWCTG. "
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
                    "baja": [-2.0, -2.0, 0.0, 2.5],
                    "confortable": [1.5, 4.0, 6.0],
                    "alta": [5.0, 8.0, 11.0],
                    "muy_alta": [9.5, 12.0, 14.0, 14.0],
                },
                mf_types={
                    "baja": "trapezoidal",
                    "muy_alta": "trapezoidal",
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
                    "baja": [0.0, 0.0, 0.12, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 0.85, 1.0, 1.0],
                },
                mf_types={
                    "baja": "trapezoidal",
                    "alta": "trapezoidal",
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
                    "baja": [0.0, 0.0, 0.12, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "alta": [0.60, 0.85, 1.0, 1.0],
                },
                mf_types={
                    "baja": "trapezoidal",
                    "alta": "trapezoidal",
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
                    "barata": [0.0, 0.0, 0.12, 0.30],
                    "media": [0.20, 0.50, 0.75],
                    "cara": [0.60, 0.85, 1.0, 1.0],
                },
                mf_types={
                    "barata": "trapezoidal",
                    "cara": "trapezoidal",
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
                "muy_baja": [0.0, 0.0, 6.0, 18.0],
                "baja": [10.0, 24.0, 38.0],
                "media": [32.0, 50.0, 66.0],
                "alta": [58.0, 76.0, 90.0],
                "muy_alta": [84.0, 95.0, 100.0, 100.0],
            },
            mf_types={
                "muy_baja": "trapezoidal",
                "muy_alta": "trapezoidal",
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
    """Dispositivo termico controlado por una salida difusa en [0, 100].

    El estado principal es la temperatura del dispositivo o del ambiente
    controlado. El modelo usa un balance termico discreto de primer orden:

        T(k+1) = T(k)
               + Delta_ambiente
               + Delta_ocupacion
               + Delta_solar
               + Delta_uso
               + Delta_control

    HVAC y refrigerador comparten esta forma general, pero el termino de
    control se interpreta de forma distinta:

    - HVAC: puede enfriar o calentar segun el error respecto a la meta.
    - Refrigerador: solo enfria.
    """

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
        self.control_mode = "idle"
        self.reset()

    def reset(self, initial_temp: Optional[float] = None):
        self.temperature = initial_temp if initial_temp is not None else self.dynamics.initial_temperature
        self.power_level = 0.0
        self.consumption_kw = 0.0
        self.control_mode = "idle"

    def step(
        self,
        ambient_temperature: float,
        occupancy: float,
        solar_radiation: float,
        usage_load: float = 0.0,
        control_level: Optional[float] = None,
    ) -> Dict[str, float]:
        """Avanza un paso temporal del modelo dinamico.

        Variables intermedias:

        - ctrl_norm = control_level / 100
        - Delta_ambiente = k_a * (T_amb - T)
        - Delta_ocupacion = k_o * occupancy
        - Delta_solar = k_s * (radiacion / 1000) * 10
        - Delta_uso = k_u * usage_load
        - Delta_control = k_c * ctrl_norm

        Para HVAC:
            se usa una banda muerta alrededor del setpoint. Fuera de ella,
            el equipo aplica la potencia pedida por el controlador; dentro
            de ella queda inactivo para evitar oscilaciones innecesarias.

        Para refrigerador:
            siempre enfria cuando hay control activo.
        """
        cfg = self.dynamics
        if control_level is None:
            control_level = usage_load
            usage_load = 0.0
        requested_ctrl_norm = float(np.clip(control_level / 100.0, 0.0, 1.0))
        effective_ctrl_norm = requested_ctrl_norm
        self.power_level = effective_ctrl_norm

        delta_ambient = cfg.ambient_coupling * self.dt * (ambient_temperature - self.temperature)
        delta_occupancy = cfg.occupancy_gain * self.dt * occupancy
        delta_solar = cfg.solar_gain * self.dt * (solar_radiation / 1000.0) * 10.0
        delta_usage = cfg.usage_gain * self.dt * usage_load
        delta_control = cfg.control_gain * self.dt * effective_ctrl_norm

        temp_error = self.temperature - self.target_temperature
        if self.descriptor.key == "hvac":
            deadband = max(self.comfort_range / 2.0, 0.15)
            effective_ctrl_norm = requested_ctrl_norm
            self.power_level = effective_ctrl_norm

            if temp_error > deadband:
                control_effect = -delta_control
                self.control_mode = "cooling"
            elif temp_error < -deadband:
                control_effect = delta_control
                self.control_mode = "heating"
            else:
                control_effect = 0.0
                self.control_mode = "idle"
        else:
            effective_ctrl_norm = requested_ctrl_norm
            self.power_level = effective_ctrl_norm
            control_effect = -delta_control
            self.control_mode = "cooling" if effective_ctrl_norm > 0.01 else "idle"

        self.temperature += delta_ambient + delta_occupancy + delta_solar + delta_usage + control_effect
        self.temperature = float(np.clip(self.temperature, cfg.min_temperature, cfg.max_temperature))

        if effective_ctrl_norm > 0.01:
            self.consumption_kw = cfg.max_power_kw * effective_ctrl_norm / max(cfg.cop, 0.1) + cfg.standby_kw
        else:
            self.consumption_kw = 0.0

        return self.get_state()

    def get_state(self) -> Dict[str, float]:
        return {
            "device_temperature": round(self.temperature, 3),
            "device_power_level": round(self.power_level, 4),
            "device_consumption_kw": round(self.consumption_kw, 4),
            "control_mode": self.control_mode,
        }

    def get_temp_error(self) -> float:
        return self.temperature - self.target_temperature

    @property
    def display_name(self) -> str:
        return self.descriptor.display_name

    @property
    def name(self) -> str:
        return self.descriptor.key
