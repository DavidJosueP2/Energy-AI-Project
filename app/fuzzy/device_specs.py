"""
Especificaciones difusas por dispositivo.

Define la estructura interpretable del sistema difuso para cada dispositivo:
- variables linguisticas de entrada y salida
- funciones de pertenencia
- reglas
- mapeos para interfaz y simulacion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


MembershipParams = List[float]


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
    """Especificacion completa del sistema difuso de un dispositivo."""

    device_key: str
    display_name: str
    control_display_name: str
    temperature_display_name: str
    output_name: str
    output_display_name: str
    target_temperature: float
    comfort_range: float
    variables: List[VariableSpec]
    output_variable: VariableSpec
    explanation: str = ""

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


def build_device_spec(device_key: str) -> DeviceFuzzySpec:
    """Construye la especificacion difusa del dispositivo solicitado."""
    if device_key == "refrigerador":
        return _build_refrigerator_spec()
    return _build_hvac_spec()


def _build_hvac_spec() -> DeviceFuzzySpec:
    return DeviceFuzzySpec(
        device_key="hvac",
        display_name="HVAC (Climatizacion)",
        control_display_name="Nivel de climatizacion",
        temperature_display_name="Temperatura interior",
        output_name="control_output",
        output_display_name="Nivel de climatizacion",
        target_temperature=22.0,
        comfort_range=2.0,
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
            display_name="Nivel de climatizacion",
            universe_range=(0.0, 100.0),
            role="output",
            description="Potencia relativa del climatizador.",
            unit="%",
            sets={
                "muy_baja": [0.0, 0.0, 20.0],
                "baja": [10.0, 25.0, 40.0],
                "media": [35.0, 52.0, 68.0],
                "alta": [60.0, 78.0, 90.0],
                "muy_alta": [82.0, 100.0, 100.0],
            },
        ),
    )


def _build_refrigerator_spec() -> DeviceFuzzySpec:
    return DeviceFuzzySpec(
        device_key="refrigerador",
        display_name="Refrigerador",
        control_display_name="Nivel de enfriamiento",
        temperature_display_name="Temperatura interna del refrigerador",
        output_name="control_output",
        output_display_name="Nivel de enfriamiento",
        target_temperature=4.0,
        comfort_range=1.5,
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
            display_name="Nivel de enfriamiento",
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
