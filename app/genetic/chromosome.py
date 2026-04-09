"""
Codificacion cromosomica de parametros de membresia.
"""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from app.config import FuzzyConfig
from app.fuzzy.controller import FuzzyController
from app.fuzzy.device_specs import DeviceFuzzySpec, build_device_spec


class ChromosomeEncoder:
    """Codifica las funciones de pertenencia del controlador activo."""

    def __init__(
        self,
        fuzzy_config: Optional[FuzzyConfig] = None,
        controller: Optional[FuzzyController] = None,
        spec: Optional[DeviceFuzzySpec] = None,
    ):
        if controller is not None:
            self.spec = deepcopy(controller.spec)
        else:
            self.spec = deepcopy(spec or build_device_spec("hvac"))

        self._variable_specs = self._build_variable_specs()
        self._gene_specs = self._build_gene_specs()
        self.chromosome_length = len(self._gene_specs)

    def _build_variable_specs(self) -> List[Dict]:
        variables = []
        for variable_spec in self.spec.input_variables:
            variables.append(
                {
                    "name": variable_spec.name,
                    "sets": variable_spec.sets,
                    "range": variable_spec.universe_range,
                    "role": "input",
                }
            )
        variables.append(
            {
                "name": self.spec.output_variable.name,
                "sets": self.spec.output_variable.sets,
                "range": self.spec.output_variable.universe_range,
                "role": "output",
            }
        )
        return variables

    def _build_gene_specs(self) -> List[Dict]:
        genes: List[Dict] = []
        for variable in self._variable_specs:
            low, high = variable["range"]
            margin = (high - low) * 0.15
            for set_name, params in variable["sets"].items():
                for param_idx, param_value in enumerate(params):
                    genes.append(
                        {
                            "variable": variable["name"],
                            "set": set_name,
                            "param_idx": param_idx,
                            "default": param_value,
                            "min": low - margin,
                            "max": high + margin,
                        }
                    )
        return genes

    def encode_default(self) -> np.ndarray:
        return np.array([gene["default"] for gene in self._gene_specs], dtype=float)

    def decode(self, chromosome: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        params: Dict[str, Dict[str, List[float]]] = {}
        for idx, gene in enumerate(self._gene_specs):
            variable_name = gene["variable"]
            set_name = gene["set"]
            if variable_name not in params:
                params[variable_name] = {}
            if set_name not in params[variable_name]:
                template_length = len(self.spec.get_variable(variable_name).sets[set_name])
                params[variable_name][set_name] = [0.0] * template_length
            params[variable_name][set_name][gene["param_idx"]] = float(chromosome[idx])
        if "control_output" in params:
            params["hvac_output"] = deepcopy(params["control_output"])
        return params

    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        repaired = chromosome.copy()

        for idx, gene in enumerate(self._gene_specs):
            repaired[idx] = np.clip(repaired[idx], gene["min"], gene["max"])

        decoded = self.decode(repaired)
        for variable_name, sets_dict in decoded.items():
            for set_name, params in sets_dict.items():
                if len(params) == 3:
                    a, b, c = sorted(params)
                    if b - a < 0.05:
                        b = a + 0.05
                    if c - b < 0.05:
                        c = b + 0.05
                    sets_dict[set_name] = [a, b, c]
                elif len(params) == 4:
                    a, b, c, d = sorted(params)
                    if b - a < 0.05:
                        b = a + 0.05
                    if c - b < 0.05:
                        c = b + 0.05
                    if d - c < 0.05:
                        d = c + 0.05
                    sets_dict[set_name] = [a, b, c, d]

        return self._encode_from_decoded(decoded)

    def _encode_from_decoded(self, decoded: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
        chromosome = np.zeros(self.chromosome_length, dtype=float)
        for idx, gene in enumerate(self._gene_specs):
            variable_name = gene["variable"]
            source_name = variable_name
            if source_name not in decoded and variable_name == "control_output" and "hvac_output" in decoded:
                source_name = "hvac_output"
            chromosome[idx] = decoded[source_name][gene["set"]][gene["param_idx"]]
        return chromosome

    def get_gene_info(self) -> List[Dict]:
        return deepcopy(self._gene_specs)

    def generate_random(self, rng: np.random.RandomState, perturbation: float = 2.0) -> np.ndarray:
        base = self.encode_default()
        noise = rng.uniform(-perturbation, perturbation, self.chromosome_length)
        for idx, gene in enumerate(self._gene_specs):
            noise[idx] *= (gene["max"] - gene["min"]) / 20.0
        return self.repair(base + noise)
