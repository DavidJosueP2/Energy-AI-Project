# ==============================================================================
# controller.py - Controlador difuso completo
# ==============================================================================
"""
Controlador difuso de alto nivel que integra:
- Variables de entrada y salida con funciones de pertenencia
- Base de reglas
- Motor de inferencia Mamdani

Proporciona una interfaz simple: recibe variables del entorno,
retorna nivel de climatizacion. Es el componente que se conecta
directamente con el simulador.

Expone el proceso interno de inferencia para visualizacion:
- Grados de pertenencia de cada variable
- Reglas activadas con sus fuerzas
- Funcion agregada
- Valor defuzzificado
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass, field

from app.config import FuzzyConfig
from app.fuzzy.membership import FuzzyVariable
from app.fuzzy.rules import RuleBase, FuzzyRule, create_default_rule_base
from app.fuzzy.inference import MamdaniInference


@dataclass
class InferenceDetail:
    """Detalle completo de un ciclo de inferencia para visualizacion."""
    crisp_inputs: Dict[str, float] = field(default_factory=dict)
    membership_degrees: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rules_with_strength: List[Tuple[FuzzyRule, float]] = field(default_factory=list)
    aggregated_output: Optional[np.ndarray] = None
    centroid_value: float = 0.0
    output_label: str = ""

    @property
    def active_rules(self) -> List[Tuple[FuzzyRule, float]]:
        """Retorna solo las reglas con fuerza de disparo > 0."""
        return [(r, s) for r, s in self.rules_with_strength if s > 0.01]

    @property
    def top_rules(self) -> List[Tuple[FuzzyRule, float]]:
        """Retorna las reglas mas influyentes ordenadas."""
        active = self.active_rules
        active.sort(key=lambda x: x[1], reverse=True)
        return active[:10]


class FuzzyController:
    """
    Controlador difuso para gestion energetica residencial.

    Entradas:
    - temp_error: diferencia entre T_interior y T_objetivo
    - humidity: humedad relativa normalizada [0, 1]
    - occupancy: numero de ocupantes
    - tariff: tarifa electrica normalizada [0, 1]
    - consumption: consumo actual normalizado [0, 1]

    Salida:
    - hvac_output: nivel de climatizacion [0, 100]
    """

    def __init__(self, config: Optional[FuzzyConfig] = None,
                 rule_base: Optional[RuleBase] = None):
        self.config = config or FuzzyConfig()
        self.rule_base = rule_base or create_default_rule_base()

        # Variables difusas
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: Optional[FuzzyVariable] = None

        # Ultimo detalle de inferencia (para visualizacion)
        self.last_inference: Optional[InferenceDetail] = None

        self._build_variables()
        self._build_inference_engine()

    def _build_variables(self):
        """Construye las variables difusas con sus funciones de pertenencia."""
        cfg = self.config
        res = cfg.universe_resolution

        # Variable: Error de temperatura
        self.input_variables['temp_error'] = FuzzyVariable(
            'temp_error', cfg.temp_error_range, res
        )
        self.input_variables['temp_error'].add_triangular_sets(cfg.temp_error_sets)

        # Variable: Humedad
        self.input_variables['humidity'] = FuzzyVariable(
            'humidity', cfg.humidity_range, res
        )
        self.input_variables['humidity'].add_triangular_sets(cfg.humidity_sets)

        # Variable: Ocupacion
        self.input_variables['occupancy'] = FuzzyVariable(
            'occupancy', cfg.occupancy_range, res
        )
        self.input_variables['occupancy'].add_triangular_sets(cfg.occupancy_sets)

        # Variable: Tarifa normalizada
        self.input_variables['tariff'] = FuzzyVariable(
            'tariff', cfg.tariff_range, res
        )
        self.input_variables['tariff'].add_triangular_sets(cfg.tariff_sets)

        # Variable: Consumo normalizado
        self.input_variables['consumption'] = FuzzyVariable(
            'consumption', cfg.consumption_range, res
        )
        self.input_variables['consumption'].add_triangular_sets(cfg.consumption_sets)

        # Variable de salida: Nivel de climatizacion
        self.output_variable = FuzzyVariable(
            'hvac_output', cfg.output_range, res
        )
        self.output_variable.add_triangular_sets(cfg.output_sets)

    def _build_inference_engine(self):
        """Construye el motor de inferencia Mamdani."""
        self.inference_engine = MamdaniInference(
            input_variables=self.input_variables,
            output_variable=self.output_variable,
            rule_base=self.rule_base,
        )

    def evaluate(self, inputs: Dict[str, float]) -> float:
        """
        Evalua el controlador difuso con las entradas dadas.
        Guarda el detalle de inferencia para posterior visualizacion.

        Args:
            inputs: Dict con variables del entorno.

        Returns:
            Nivel de climatizacion [0, 100].
        """
        # Mapear nombres del simulador a nombres del controlador
        controller_inputs = {
            'temp_error': inputs.get('temp_error', 0.0),
            'humidity': inputs.get('humidity', 0.5),
            'occupancy': inputs.get('occupancy', 0.0),
            'tariff': inputs.get('tariff_normalized', inputs.get('tariff', 0.5)),
            'consumption': inputs.get('consumption_normalized', inputs.get('consumption', 0.5)),
        }

        # Inferencia difusa con detalle completo
        output, detail = self.evaluate_with_detail(controller_inputs)

        return float(np.clip(output, 0.0, 100.0))

    def evaluate_with_detail(self, controller_inputs: Dict[str, float]
                              ) -> Tuple[float, InferenceDetail]:
        """
        Evalua el controlador y retorna detalle completo de la inferencia.
        Util para la pestana de visualizacion difusa.

        Args:
            controller_inputs: Dict con valores de las variables de entrada.

        Returns:
            Tupla (valor_salida, InferenceDetail).
        """
        detail = InferenceDetail()
        detail.crisp_inputs = dict(controller_inputs)

        # 1. Fuzzificacion
        detail.membership_degrees = self.inference_engine._fuzzify(controller_inputs)

        # 2-3. Evaluacion de reglas
        activated_outputs = self.inference_engine._evaluate_rules(detail.membership_degrees)
        detail.rules_with_strength = self.inference_engine.get_rule_activations(controller_inputs)

        # 4. Agregacion
        detail.aggregated_output = self.inference_engine._aggregate(activated_outputs)

        # 5. Desfuzzificacion
        detail.centroid_value = self.inference_engine._defuzzify(detail.aggregated_output)

        # Clasificar salida
        val = detail.centroid_value
        if val < 15:
            detail.output_label = 'Muy Baja'
        elif val < 35:
            detail.output_label = 'Baja'
        elif val < 60:
            detail.output_label = 'Media'
        elif val < 80:
            detail.output_label = 'Alta'
        else:
            detail.output_label = 'Muy Alta'

        # Guardar para acceso posterior
        self.last_inference = detail

        return detail.centroid_value, detail

    def get_controller_function(self):
        """Retorna una funcion callable para pasar al simulador."""
        return self.evaluate

    def get_membership_params(self) -> Dict[str, Dict[str, List[float]]]:
        """Retorna todos los parametros de funciones de pertenencia."""
        params = {}
        for var_name, variable in self.input_variables.items():
            params[var_name] = {}
            for set_name, fuzzy_set in variable.sets.items():
                params[var_name][set_name] = list(fuzzy_set.params)

        params['hvac_output'] = {}
        for set_name, fuzzy_set in self.output_variable.sets.items():
            params['hvac_output'][set_name] = list(fuzzy_set.params)

        return params

    def set_membership_params(self, params: Dict[str, Dict[str, List[float]]]):
        """Establece parametros de funciones de pertenencia (usado por GA)."""
        for var_name, sets_dict in params.items():
            if var_name == 'hvac_output':
                variable = self.output_variable
            elif var_name in self.input_variables:
                variable = self.input_variables[var_name]
            else:
                continue

            for set_name, new_params in sets_dict.items():
                if set_name in variable.sets:
                    variable.sets[set_name].params = list(new_params)

        self._build_inference_engine()

    def clone(self) -> 'FuzzyController':
        """Crea una copia profunda del controlador."""
        new_config = deepcopy(self.config)
        new_rb = deepcopy(self.rule_base)
        new_controller = FuzzyController(new_config, new_rb)
        new_controller.set_membership_params(self.get_membership_params())
        return new_controller

    def get_variable_info(self) -> Dict[str, dict]:
        """Retorna informacion de todas las variables para visualizacion."""
        info = {}
        for name, var in self.input_variables.items():
            info[name] = {
                'range': var.universe_range,
                'sets': {s: fs.params for s, fs in var.sets.items()},
                'type': 'input',
            }
        info['hvac_output'] = {
            'range': self.output_variable.universe_range,
            'sets': {s: fs.params for s, fs in self.output_variable.sets.items()},
            'type': 'output',
        }
        return info

    def __repr__(self) -> str:
        n_inputs = len(self.input_variables)
        n_rules = self.rule_base.num_rules
        return f"FuzzyController({n_inputs} inputs, {n_rules} rules)"
