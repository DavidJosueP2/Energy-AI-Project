from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import FuzzyConfig
from app.fuzzy.inference import MamdaniInference
from app.fuzzy.membership import FuzzyVariable
from app.fuzzy.rules import FuzzyRule, RuleSet, create_default_rule_base
from app.simulation.devices import DeviceFuzzySpec, VariableSpec, build_device_spec


@dataclass
class ConsequentActivation:
    """Representa un consecuente activado y recortado por una regla."""

    set_name: str
    strength: float
    source_rule: str


@dataclass
class InferenceDetail:
    """Traza completa de una inferencia puntual."""

    crisp_inputs: Dict[str, float] = field(default_factory=dict)
    membership_degrees: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rules_with_strength: List[Tuple[FuzzyRule, float]] = field(default_factory=list)
    consequent_activations: List[ConsequentActivation] = field(default_factory=list)
    aggregated_output: Optional[np.ndarray] = None
    centroid_value: float = 0.0
    output_label: str = ""
    output_memberships: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    @property
    def active_rules(self) -> List[Tuple[FuzzyRule, float]]:
        return [(rule, strength) for rule, strength in self.rules_with_strength if strength > 0.01]

    @property
    def top_rules(self) -> List[Tuple[FuzzyRule, float]]:
        active = self.active_rules
        active.sort(key=lambda item: item[1], reverse=True)
        return active[:10]


class FuzzyController:
    """
    Controlador Mamdani explicable para HVAC o refrigerador.

    Mantiene compatibilidad parcial con la firma anterior, pero ahora
    el comportamiento real queda definido por `DeviceFuzzySpec`.
    """

    def __init__(
        self,
        config: Optional[FuzzyConfig] = None,
        rule_base: Optional[RuleSet] = None,
        device_key: str = "hvac",
        spec: Optional[DeviceFuzzySpec] = None,
    ):
        self.legacy_config = config or FuzzyConfig()
        self.spec = spec or build_device_spec(device_key)
        self.device_key = self.spec.device_key
        self.rule_base = rule_base or create_default_rule_base(
            device_key=self.device_key,
            output_name=self.spec.output_name,
        )

        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: Optional[FuzzyVariable] = None
        self.last_inference: Optional[InferenceDetail] = None

        self._build_variables()
        self._build_inference_engine()

    def _build_variables(self):
        resolution = self.legacy_config.universe_resolution
        self.input_variables = {}
        for variable_spec in self.spec.input_variables:
            variable = FuzzyVariable(
                variable_spec.name,
                variable_spec.universe_range,
                resolution,
            )
            for set_name, params in variable_spec.sets.items():
                variable.add_set(set_name, variable_spec.get_mf_type(set_name), params)
            self.input_variables[variable_spec.name] = variable

        output_spec = self.spec.output_variable
        self.output_variable = FuzzyVariable(
            output_spec.name,
            output_spec.universe_range,
            resolution,
        )
        for set_name, params in output_spec.sets.items():
            self.output_variable.add_set(set_name, output_spec.get_mf_type(set_name), params)

    def _build_inference_engine(self):
        self.inference_engine = MamdaniInference(
            input_variables=self.input_variables,
            output_variable=self.output_variable,
            fuzzy_rule_set=self.rule_base,
        )

    def normalize_inputs(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Mapea entradas de simulacion o interfaz a las variables del controlador."""
        normalized: Dict[str, float] = {}
        for variable_spec in self.spec.input_variables:
            source_key = variable_spec.input_key or variable_spec.name
            if variable_spec.name in inputs:
                normalized[variable_spec.name] = inputs[variable_spec.name]
            elif source_key in inputs:
                normalized[variable_spec.name] = inputs[source_key]
            elif variable_spec.name == "tariff" and "tariff" in inputs:
                normalized[variable_spec.name] = inputs["tariff"]

        self._apply_runtime_adjustments(normalized, inputs)
        return normalized

    def _apply_runtime_adjustments(self, normalized: Dict[str, float], raw_inputs: Dict[str, float]):
        """Ajusta entradas segun el contexto runtime del dispositivo."""
        def bounded_comfort_factor(
            reference_range: float,
            runtime_range: float,
            min_factor: float,
            max_factor: float,
        ) -> float:
            """Modula sensibilidad sin volver inestable al controlador.

            Un rango de confort menor puede volver la decision un poco mas
            exigente, pero no debe multiplicar la sensibilidad de forma
            explosiva. Usamos la raiz cuadrada de la razon y la acotamos
            para preservar interpretabilidad y evitar sobre-reaccion.
            """
            ratio = max(reference_range, 1e-6) / max(runtime_range, 1e-6)
            return float(np.clip(np.sqrt(ratio), min_factor, max_factor))

        if self.device_key == "refrigerador" and "device_temperature" in normalized:
            reference_target = float(self.spec.descriptor.default_target_temperature)
            reference_range = max(float(self.spec.descriptor.default_comfort_range), 1e-6)
            runtime_target = float(raw_inputs.get("target_temperature", reference_target))
            runtime_comfort = max(float(raw_inputs.get("comfort_range", reference_range)), 1e-6)
            raw_temperature = float(normalized["device_temperature"])
            base_shift_gain = 2.5
            comfort_factor = bounded_comfort_factor(
                reference_range=reference_range,
                runtime_range=runtime_comfort,
                min_factor=0.90,
                max_factor=1.20,
            )
            shift_gain = base_shift_gain * comfort_factor
            shifted_temperature = reference_target + shift_gain * (raw_temperature - runtime_target)

            temp_spec = self.spec.get_variable("device_temperature")
            low, high = temp_spec.universe_range
            normalized["device_temperature"] = float(np.clip(shifted_temperature, low, high))
            normalized["target_temperature"] = runtime_target
            normalized["comfort_range"] = runtime_comfort
            normalized["raw_device_temperature"] = raw_temperature
            return

        if self.device_key != "hvac" or "temp_error" not in normalized:
            return

        reference_range = max(self.spec.descriptor.default_comfort_range, 1e-6)
        comfort_range = max(float(raw_inputs.get("comfort_range", reference_range)), 1e-6)
        raw_error = float(raw_inputs.get("raw_temp_error", normalized["temp_error"]))
        control_error = float(normalized["temp_error"])
        comfort_factor = bounded_comfort_factor(
            reference_range=reference_range,
            runtime_range=comfort_range,
            min_factor=0.85,
            max_factor=1.35,
        )
        scaled_error = control_error * comfort_factor

        temp_error_spec = self.spec.get_variable("temp_error")
        low, high = temp_error_spec.universe_range
        normalized["temp_error"] = float(np.clip(scaled_error, low, high))
        normalized["comfort_range"] = comfort_range
        normalized["raw_temp_error"] = raw_error

    def evaluate(self, inputs: Dict[str, float]) -> float:
        controller_inputs = self.normalize_inputs(inputs)
        output, _ = self.evaluate_with_detail(controller_inputs)
        low, high = self.output_variable.universe_range
        return float(np.clip(output, low, high))

    def evaluate_with_detail(self, controller_inputs: Dict[str, float]) -> Tuple[float, InferenceDetail]:
        controller_inputs = self.normalize_inputs(controller_inputs)
        detail = InferenceDetail(crisp_inputs=dict(controller_inputs))

        detail.membership_degrees = self.inference_engine._fuzzify(controller_inputs)
        activated_outputs = self.inference_engine._evaluate_rules(detail.membership_degrees)
        implied_outputs = self.inference_engine._implicate(activated_outputs)
        detail.rules_with_strength = self.inference_engine.get_rule_activations(controller_inputs)
        detail.consequent_activations = [
            ConsequentActivation(
                set_name=rule.consequent[1],
                strength=strength,
                source_rule=str(rule),
            )
            for rule, strength in detail.rules_with_strength
            if strength > 0.01
        ]
        
        detail.aggregated_output = self.inference_engine._aggregate(implied_outputs)
        detail.centroid_value = self.inference_engine._defuzzify(detail.aggregated_output)
        detail.output_memberships = self.output_variable.fuzzify(detail.centroid_value)
        detail.output_label = max(
            detail.output_memberships.items(),
            key=lambda item: item[1],
        )[0]
        detail.explanation = self._build_explanation(detail)
        self.last_inference = detail
        return detail.centroid_value, detail

    def _build_explanation(self, detail: InferenceDetail) -> str:
        if not detail.top_rules:
            return "No se activaron reglas con peso significativo."

        rule, strength = detail.top_rules[0]
        readable_inputs = ", ".join(
            f"{name}={value:.2f}" for name, value in detail.crisp_inputs.items()
            if name != "raw_temp_error"
        )
        comfort_note = ""
        raw_error = detail.crisp_inputs.get("raw_temp_error")
        comfort_range = detail.crisp_inputs.get("comfort_range")
        scaled_error = detail.crisp_inputs.get("temp_error")
        if self.device_key == "hvac" and raw_error is not None and comfort_range is not None and scaled_error is not None:
            comfort_note = (
                f" El error termico bruto fue {raw_error:.2f} y se ajusto a "
                f"{scaled_error:.2f} usando rango de confort {comfort_range:.2f}."
            )
        elif self.device_key == "refrigerador":
            raw_temperature = detail.crisp_inputs.get("raw_device_temperature")
            runtime_target = detail.crisp_inputs.get("target_temperature")
            shifted_temperature = detail.crisp_inputs.get("device_temperature")
            comfort_range = detail.crisp_inputs.get("comfort_range")
            if raw_temperature is not None and runtime_target is not None and shifted_temperature is not None:
                comfort_note = (
                    f" La temperatura interna real fue {raw_temperature:.2f} C y se "
                    f"reexpreso como {shifted_temperature:.2f} C respecto al setpoint "
                    f"runtime {runtime_target:.2f} C"
                )
                if comfort_range is not None:
                    comfort_note += f" usando rango de confort {comfort_range:.2f} C."
                else:
                    comfort_note += "."
        return (
            f"Entradas evaluadas: {readable_inputs}. "
            f"La regla dominante fue '{rule.description or str(rule)}' "
            f"con activacion {strength:.3f}. "
            f"La salida dominante es '{detail.output_label.replace('_', ' ')}' "
            f"con valor defuzzificado {detail.centroid_value:.2f}."
            f"{comfort_note}"
        )

    def get_controller_function(self):
        return self.evaluate

    def get_membership_params(self) -> Dict[str, Dict[str, List[float]]]:
        params: Dict[str, Dict[str, List[float]]] = {}
        for variable_name, variable in self.input_variables.items():
            params[variable_name] = {
                set_name: list(fuzzy_set.params)
                for set_name, fuzzy_set in variable.sets.items()
            }
        params[self.output_variable.name] = {
            set_name: list(fuzzy_set.params)
            for set_name, fuzzy_set in self.output_variable.sets.items()
        }
        if self.output_variable.name == "control_output":
            params["hvac_output"] = deepcopy(params[self.output_variable.name])
        return params

    def set_membership_params(self, params: Dict[str, Dict[str, List[float]]]):
        for variable_name, sets_dict in params.items():
            if variable_name == self.output_variable.name or (
                variable_name == "hvac_output" and self.output_variable.name == "control_output"
            ):
                variable = self.output_variable
            else:
                variable = self.input_variables.get(variable_name)

            if variable is None:
                continue

            for set_name, new_params in sets_dict.items():
                if set_name in variable.sets:
                    variable.sets[set_name].params = list(new_params)

        self._sync_spec_from_variables()
        self._build_inference_engine()

    def _sync_spec_from_variables(self):
        for variable_spec in self.spec.input_variables:
            variable = self.input_variables[variable_spec.name]
            variable_spec.sets = {
                set_name: list(fuzzy_set.params)
                for set_name, fuzzy_set in variable.sets.items()
            }

        self.spec.output_variable.sets = {
            set_name: list(fuzzy_set.params)
            for set_name, fuzzy_set in self.output_variable.sets.items()
        }

    def clone(self) -> "FuzzyController":
        new_spec = deepcopy(self.spec)
        new_rule_base = deepcopy(self.rule_base)
        new_controller = FuzzyController(
            config=deepcopy(self.legacy_config),
            rule_base=new_rule_base,
            device_key=self.device_key,
            spec=new_spec,
        )
        new_controller.set_membership_params(self.get_membership_params())
        return new_controller

    def get_variable_info(self) -> Dict[str, dict]:
        info: Dict[str, dict] = {}
        for variable_spec in self.spec.input_variables:
            info[variable_spec.name] = {
                "range": variable_spec.universe_range,
                "sets": deepcopy(variable_spec.sets),
                "type": "input",
                "display_name": variable_spec.display_name,
                "unit": variable_spec.unit,
            }
        info[self.spec.output_variable.name] = {
            "range": self.spec.output_variable.universe_range,
            "sets": deepcopy(self.spec.output_variable.sets),
            "type": "output",
            "display_name": self.spec.output_variable.display_name,
            "unit": self.spec.output_variable.unit,
        }
        return info

    @property
    def output_name(self) -> str:
        return self.spec.output_name

    def __repr__(self) -> str:
        return (
            f"FuzzyController(device={self.spec.display_name}, "
            f"inputs={len(self.input_variables)}, rules={self.rule_base.num_rules})"
        )
