# ==============================================================================
# controller.py - Controlador difuso completo
# ==============================================================================
"""
Controlador difuso de alto nivel que integra:
- Variables de entrada y salida con funciones de pertenencia
- Base de reglas
- Motor de inferencia Mamdani

Proporciona una interfaz simple: recibe variables del entorno,
retorna nivel de climatización. Es el componente que se conecta
directamente con el simulador.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from app.config import FuzzyConfig
from app.fuzzy.membership import FuzzyVariable
from app.fuzzy.rules import RuleBase, create_default_rule_base
from app.fuzzy.inference import MamdaniInference


class FuzzyController:
    """
    Controlador difuso para gestión energética residencial.
    
    Entradas:
    - temp_error: diferencia entre T_interior y T_objetivo (°C)
    - occupancy: número de ocupantes
    - tariff: tarifa eléctrica normalizada [0, 1]
    - consumption: consumo actual normalizado [0, 1]
    
    Salida:
    - hvac_output: nivel de climatización [0, 100]
    """
    
    def __init__(self, config: Optional[FuzzyConfig] = None,
                 rule_base: Optional[RuleBase] = None):
        """
        Args:
            config: Configuración del controlador difuso.
            rule_base: Base de reglas. Si None, usa la base por defecto.
        """
        self.config = config or FuzzyConfig()
        self.rule_base = rule_base or create_default_rule_base()
        
        # Crear variables difusas
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: Optional[FuzzyVariable] = None
        
        self._build_variables()
        self._build_inference_engine()
    
    def _build_variables(self):
        """Construye las variables difusas con sus funciones de pertenencia."""
        cfg = self.config
        res = cfg.universe_resolution
        
        # --- Variable: Error de temperatura ---
        self.input_variables['temp_error'] = FuzzyVariable(
            'temp_error', cfg.temp_error_range, res
        )
        self.input_variables['temp_error'].add_triangular_sets(cfg.temp_error_sets)
        
        # --- Variable: Ocupación ---
        self.input_variables['occupancy'] = FuzzyVariable(
            'occupancy', cfg.occupancy_range, res
        )
        self.input_variables['occupancy'].add_triangular_sets(cfg.occupancy_sets)
        
        # --- Variable: Tarifa normalizada ---
        self.input_variables['tariff'] = FuzzyVariable(
            'tariff', cfg.tariff_range, res
        )
        self.input_variables['tariff'].add_triangular_sets(cfg.tariff_sets)
        
        # --- Variable: Consumo normalizado ---
        self.input_variables['consumption'] = FuzzyVariable(
            'consumption', cfg.consumption_range, res
        )
        self.input_variables['consumption'].add_triangular_sets(cfg.consumption_sets)
        
        # --- Variable de salida: Nivel de climatización ---
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
        Evalúa el controlador difuso con las entradas dadas.
        
        Esta es la función principal que se pasa al simulador.
        Mapea las variables del simulador a las entradas del controlador.
        
        Args:
            inputs: Dict con variables del entorno:
                - temp_error: T_interior - T_objetivo
                - occupancy: número de ocupantes
                - tariff_normalized: tarifa [0, 1]
                - consumption_normalized: consumo [0, 1]
                
        Returns:
            Nivel de climatización [0, 100].
        """
        # Mapear nombres del simulador a nombres del controlador
        controller_inputs = {
            'temp_error': inputs.get('temp_error', 0.0),
            'occupancy': inputs.get('occupancy', 0.0),
            'tariff': inputs.get('tariff_normalized', 0.5),
            'consumption': inputs.get('consumption_normalized', 0.5),
        }
        
        # Inferencia difusa
        output = self.inference_engine.infer(controller_inputs)
        
        # Asegurar rango válido
        return float(np.clip(output, 0.0, 100.0))
    
    def get_controller_function(self):
        """Retorna una función callable para pasar al simulador."""
        return self.evaluate
    
    def get_membership_params(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Retorna todos los parámetros de funciones de pertenencia.
        Útil para serialización y para el algoritmo genético.
        """
        params = {}
        for var_name, variable in self.input_variables.items():
            params[var_name] = {}
            for set_name, fuzzy_set in variable.sets.items():
                params[var_name][set_name] = list(fuzzy_set.params)
        
        # También salida
        params['hvac_output'] = {}
        for set_name, fuzzy_set in self.output_variable.sets.items():
            params['hvac_output'][set_name] = list(fuzzy_set.params)
        
        return params
    
    def set_membership_params(self, params: Dict[str, Dict[str, List[float]]]):
        """
        Establece parámetros de funciones de pertenencia.
        Usado por el algoritmo genético para aplicar cromosomas decodificados.
        """
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
        
        # Reconstruir motor de inferencia con parámetros actualizados
        self._build_inference_engine()
    
    def clone(self) -> 'FuzzyController':
        """Crea una copia profunda del controlador."""
        new_config = deepcopy(self.config)
        new_rb = deepcopy(self.rule_base)
        new_controller = FuzzyController(new_config, new_rb)
        # Copiar parámetros actuales
        new_controller.set_membership_params(self.get_membership_params())
        return new_controller
    
    def get_variable_info(self) -> Dict[str, dict]:
        """Retorna información de todas las variables para visualización."""
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
