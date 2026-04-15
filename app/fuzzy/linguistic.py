from typing import Dict, List, Optional

from app.fuzzy.controller import FuzzyController
from app.simulation.devices import DeviceFuzzySpec, build_device_spec


class LinguisticInput:
    """Traduce etiquetas linguisticas a valores representativos para un dispositivo."""

    def __init__(self, spec: Optional[DeviceFuzzySpec] = None):
        self.spec = spec or build_device_spec("hvac")
        self._aliases = {
            "temperatura": "temp_error",
            "humedad": "humidity",
            "ocupacion": "occupancy",
            "tarifa": "tariff",
            "consumo": "consumption",
            "apertura": "door_openings",
            "carga": "load_level",
        }
        self._mappings = self._build_default_mappings()

    def set_spec(self, spec: DeviceFuzzySpec):
        self.spec = spec
        self._mappings = self._build_default_mappings()

    def _build_default_mappings(self) -> Dict[str, Dict[str, float]]:
        mappings: Dict[str, Dict[str, float]] = {}
        for variable_spec in self.spec.input_variables:
            mappings[variable_spec.name] = {}
            for label, params in variable_spec.sets.items():
                if len(params) >= 3:
                    mappings[variable_spec.name][label] = params[1]
                else:
                    mappings[variable_spec.name][label] = sum(params) / len(params)
        return mappings

    def get_variables(self) -> List[str]:
        return [variable_spec.name for variable_spec in self.spec.input_variables]

    def get_display_name(self, variable_name: str) -> str:
        return self.spec.get_variable(self._resolve_variable_name(variable_name)).display_name

    def get_labels(self, variable_name: str) -> List[str]:
        return list(self._mappings.get(self._resolve_variable_name(variable_name), {}).keys())

    def to_crisp(self, variable_name: str, label: str) -> float:
        return self._mappings[self._resolve_variable_name(variable_name)][label]

    def to_controller_inputs(self, selections: Dict[str, str]) -> Dict[str, float]:
        return {
            self._resolve_variable_name(variable_name): self.to_crisp(variable_name, label)
            for variable_name, label in selections.items()
        }

    def update_mapping(self, variable_name: str, label: str, value: float):
        resolved_name = self._resolve_variable_name(variable_name)
        if resolved_name in self._mappings and label in self._mappings[resolved_name]:
            self._mappings[resolved_name][label] = value

    def _resolve_variable_name(self, variable_name: str) -> str:
        return self._aliases.get(variable_name, variable_name)


class LinguisticOutput:
    """Interpreta una salida numerica usando la variable de salida difusa."""

    def __init__(self, controller: Optional[FuzzyController] = None, spec: Optional[DeviceFuzzySpec] = None):
        self.controller = controller
        self.spec = spec or (controller.spec if controller else build_device_spec("hvac"))

    def set_controller(self, controller: FuzzyController):
        self.controller = controller
        self.spec = controller.spec

    def classify(self, value: float) -> str:
        if self.controller is None:
            labels = list(self.spec.output_variable.sets.keys())
            return labels[len(labels) // 2]
        memberships = self.controller.output_variable.fuzzify(value)
        return max(memberships.items(), key=lambda item: item[1])[0]

    def get_dual_output(self, value: float) -> Dict[str, object]:
        label = self.classify(value)
        memberships = (
            self.controller.output_variable.fuzzify(value)
            if self.controller is not None
            else {label: 1.0}
        )
        dominant_strength = memberships.get(label, 1.0)
        low, high = self.spec.output_variable.universe_range
        normalized = (value - low) / max(high - low, 1e-6)
        return {
            "valor_numerico": round(value, 2),
            "porcentaje": f"{value:.1f}%",
            "etiqueta": label,
            "etiqueta_display": label.replace("_", " ").title(),
            "descripcion": self.spec.output_display_name,
            "valor_normalizado": round(normalized, 4),
            "dominancia": round(dominant_strength, 4),
        }

    def get_all_labels(self) -> List[str]:
        return list(self.spec.output_variable.sets.keys())
