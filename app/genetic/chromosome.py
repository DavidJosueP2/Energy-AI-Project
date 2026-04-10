"""
Codificacion cromosomica de parametros de membresia.
"""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from app.fuzzy.controller import FuzzyController
from app.simulation.devices import DeviceFuzzySpec, build_device_spec


class ChromosomeEncoder:
    """Codifica las funciones de pertenencia del controlador activo."""

    def __init__(
        self,
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
        decoded.pop("hvac_output", None)
        for variable_spec in self._variable_specs:
            variable_name = variable_spec["name"]
            self._repair_variable_family(variable_spec, decoded[variable_name])

        return self._encode_from_decoded(decoded)

    def _repair_variable_family(self, variable_spec: Dict, sets_dict: Dict[str, List[float]]) -> None:
        """Mantiene conjuntos ordenados y sin huecos muertos entre vecinos."""
        low, high = variable_spec["range"]
        min_sep = max((high - low) * 0.01, 0.01)
        min_overlap = max((high - low) * 0.03, min_sep)
        ordered_names = list(variable_spec["sets"].keys())

        for set_name in ordered_names:
            params = sets_dict[set_name]
            sets_dict[set_name] = self._normalize_set(params, low, high, min_sep)

        for prev_name, curr_name in zip(ordered_names, ordered_names[1:]):
            prev_params = sets_dict[prev_name]
            curr_params = sets_dict[curr_name]
            prev_end = prev_params[-1]
            curr_start = curr_params[0]

            if curr_start > prev_end - min_overlap:
                midpoint = (prev_end + curr_start) / 2.0
                prev_cap = high if len(prev_params) == 3 else prev_params[-1]
                curr_floor = low if len(curr_params) == 3 else curr_params[0]

                new_prev_end = min(prev_cap, max(prev_params[-2] + min_sep, midpoint + min_overlap / 2.0))
                new_curr_start = max(curr_floor, min(curr_params[1] - min_sep, midpoint - min_overlap / 2.0))

                if new_curr_start > new_prev_end:
                    junction = min(
                        max((prev_params[-2] + curr_params[1]) / 2.0, prev_params[-2] + min_sep),
                        curr_params[1] - min_sep,
                    )
                    new_prev_end = junction
                    new_curr_start = junction

                prev_params[-1] = new_prev_end
                curr_params[0] = new_curr_start
                sets_dict[prev_name] = self._normalize_set(prev_params, low, high, min_sep)
                sets_dict[curr_name] = self._normalize_set(curr_params, low, high, min_sep)

        first_name = ordered_names[0]
        last_name = ordered_names[-1]
        first_template = variable_spec["sets"][first_name]
        last_template = variable_spec["sets"][last_name]
        first = sets_dict[first_name]
        last = sets_dict[last_name]

        first[0] = low
        last[-1] = high

        if len(first_template) == 3 and first_template[0] == first_template[1]:
            first[0] = low
            first[1] = low
            first[2] = max(first[2], low + min_sep)

        if len(last_template) == 3 and last_template[1] == last_template[2]:
            last[0] = min(last[0], high - min_sep)
            last[1] = high
            last[2] = high

        sets_dict[first_name] = first
        sets_dict[last_name] = last

    @staticmethod
    def _normalize_set(params: List[float], low: float, high: float, min_sep: float) -> List[float]:
        """Ordena y acota los parametros de una MF sin perder su tipo."""
        clipped = np.clip(np.array(params, dtype=float), low, high)
        ordered = np.sort(clipped)

        if len(ordered) == 3:
            a = min(ordered[0], high - 2 * min_sep)
            b = min(max(ordered[1], a + min_sep), high - min_sep)
            c = max(ordered[2], b + min_sep)
            c = min(c, high)
            b = min(b, c - min_sep)
            a = min(a, b - min_sep)
            a = max(a, low)
            return [float(a), float(b), float(c)]

        if len(ordered) == 4:
            a = min(ordered[0], high - 3 * min_sep)
            b = min(max(ordered[1], a + min_sep), high - 2 * min_sep)
            c = min(max(ordered[2], b + min_sep), high - min_sep)
            d = max(ordered[3], c + min_sep)
            d = min(d, high)
            c = min(c, d - min_sep)
            b = min(b, c - min_sep)
            a = min(a, b - min_sep)
            a = max(a, low)
            return [float(a), float(b), float(c), float(d)]

        return [float(value) for value in ordered]

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
