# ==============================================================================
# inference.py - Motor de inferencia difusa tipo Mamdani
# ==============================================================================
"""
Implementa el ciclo completo de inferencia difusa:
1. Fuzzificación: convierte valores crisp a grados de pertenencia
2. Evaluación de reglas: aplica las reglas con operador AND (mínimo)
3. Activación: recorta los consecuentes según la fuerza de cada regla
4. Agregación: combina todos los consecuentes activados (máximo)
5. Desfuzzificación: convierte la función agregada a un valor crisp (centroide)

Este motor implementa inferencia tipo Mamdani, el más utilizado
en control difuso por su interpretabilidad y capacidad de capturar
conocimiento experto.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from app.fuzzy.membership import FuzzyVariable, FuzzySet
from app.fuzzy.rules import FuzzyRule, RuleSet


class MamdaniInference:
    """
    Motor de inferencia difusa Mamdani.
    
    Operaciones:
    - AND (conjunción): operador mínimo
    - OR (disyunción): operador máximo
    - Implicación: operador mínimo (recorte)
    - Agregación: operador máximo
    - Desfuzzificación: centroide (centro de gravedad)
    """
    
    def __init__(self, 
                 input_variables: Dict[str, FuzzyVariable],
                 output_variable: FuzzyVariable,
                 rule_base: RuleSet):
        """
        Args:
            input_variables: Diccionario {nombre: FuzzyVariable} de entradas.
            output_variable: Variable difusa de salida.
            rule_base: Base de reglas a aplicar.
        """
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.rule_base = rule_base
    
    def infer(self, crisp_inputs: Dict[str, float]) -> float:
        """
        Ejecuta el ciclo completo de inferencia.
        
        Args:
            crisp_inputs: Valores crisp de las variables de entrada.
                         Ejemplo: {'temp_error': 3.5, 'occupancy': 2.0, ...}
        
        Returns:
            Valor crisp de la salida (nivel de climatización [0, 100]).
        """
        # 1. Fuzzificación
        membership_degrees = self._fuzzify(crisp_inputs)
        
        # 2-3. Evaluación de reglas y activación de consecuentes
        activated_outputs = self._evaluate_rules(membership_degrees)
        
        # 4. Agregación
        aggregated = self._aggregate(activated_outputs)
        
        # 5. Desfuzzificación
        crisp_output = self._defuzzify(aggregated)
        
        return crisp_output
    
    def _fuzzify(self, crisp_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Paso 1: Fuzzificación.
        Convierte cada valor crisp de entrada a grados de pertenencia
        en todos los conjuntos difusos de su variable.
        
        Returns:
            {nombre_variable: {nombre_conjunto: grado_pertenencia}}
        """
        memberships = {}
        for var_name, variable in self.input_variables.items():
            if var_name in crisp_inputs:
                value = crisp_inputs[var_name]
                memberships[var_name] = variable.fuzzify(value)
            else:
                # Si no se proporciona, todos los grados son 0
                memberships[var_name] = {s: 0.0 for s in variable.sets}
        return memberships
    
    def _evaluate_rules(self, 
                        memberships: Dict[str, Dict[str, float]]
                        ) -> List[Tuple[str, float]]:
        """
        Paso 2-3: Evaluación de reglas y activación.
        
        Para cada regla:
        - Calcular la fuerza de disparo (AND = mínimo de antecedentes)
        - Aplicar peso de la regla
        - Registrar el consecuente activado con su fuerza
        
        Returns:
            Lista de (nombre_conjunto_salida, fuerza_de_disparo).
        """
        activated = []
        
        for rule in self.rule_base.get_rules():
            # Calcular fuerza de disparo (AND = mínimo)
            firing_strength = self._compute_firing_strength(rule, memberships)
            
            # Aplicar peso de la regla
            firing_strength *= rule.weight
            
            # Solo considerar reglas con fuerza > 0
            if firing_strength > 1e-10:
                output_set_name = rule.consequent[1]
                activated.append((output_set_name, firing_strength))
        
        return activated
    
    def _compute_firing_strength(self, 
                                  rule: FuzzyRule,
                                  memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Calcula la fuerza de disparo de una regla.
        Operador AND: mínimo de todos los grados de pertenencia de los antecedentes.
        """
        strengths = []
        
        for var_name, set_name in rule.antecedents:
            if var_name in memberships and set_name in memberships[var_name]:
                strengths.append(memberships[var_name][set_name])
            else:
                # Si la variable o conjunto no existe, fuerza = 0
                return 0.0
        
        if not strengths:
            return 0.0
        
        # AND = mínimo
        return min(strengths)
    
    def _aggregate(self, 
                   activated_outputs: List[Tuple[str, float]]) -> np.ndarray:
        """
        Paso 4: Agregación.
        Combina todos los consecuentes activados usando operador máximo.
        
        Para cada punto del universo de salida, toma el máximo de todos
        los consecuentes recortados por su fuerza de disparo.
        
        Returns:
            Array con la función de pertenencia agregada sobre el universo de salida.
        """
        universe = self.output_variable.universe
        aggregated = np.zeros_like(universe)
        
        if not activated_outputs:
            return aggregated
        
        for set_name, strength in activated_outputs:
            if set_name in self.output_variable.sets:
                output_set = self.output_variable.sets[set_name]
                # Evaluar la función de pertenencia del consecuente
                mf_values = output_set.evaluate(universe)
                # Recortar por la fuerza de disparo (implicación por mínimo)
                clipped = np.minimum(mf_values, strength)
                # Agregar por máximo
                aggregated = np.maximum(aggregated, clipped)
        
        return aggregated
    
    def _defuzzify(self, aggregated: np.ndarray) -> float:
        """
        Paso 5: Desfuzzificación por centroide (centro de gravedad).
        
        centroide = ∫ x·μ(x) dx / ∫ μ(x) dx
        
        Numéricamente: sum(x * μ(x)) / sum(μ(x))
        
        Returns:
            Valor crisp de la salida. Si el área es cero, retorna el
            centro del universo de discurso.
        """
        universe = self.output_variable.universe
        total_area = np.sum(aggregated)
        
        if total_area < 1e-10:
            # Sin activación: retornar valor por defecto (centro del rango)
            return (self.output_variable.universe_range[0] + 
                    self.output_variable.universe_range[1]) / 2.0
        
        centroid = np.sum(universe * aggregated) / total_area
        return float(centroid)
    
    def get_rule_activations(self, 
                              crisp_inputs: Dict[str, float]
                              ) -> List[Tuple[FuzzyRule, float]]:
        """
        Retorna la lista de reglas con sus fuerzas de disparo.
        Útil para análisis y debugging del controlador.
        """
        memberships = self._fuzzify(crisp_inputs)
        results = []
        
        for rule in self.rule_base.get_rules():
            strength = self._compute_firing_strength(rule, memberships)
            strength *= rule.weight
            results.append((rule, strength))
        
        return results
