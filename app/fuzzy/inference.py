# ==============================================================================
# inference.py - Motor de inferencia difusa tipo Mamdani
# ==============================================================================
"""
Motor de inferencia difusa tipo Mamdani.

Este módulo implementa el núcleo matemático del controlador difuso del
proyecto. Su responsabilidad no es decidir *qué* variables existen ni *qué*
reglas deben definirse; su responsabilidad es ejecutar correctamente el
proceso de inferencia una vez que ya se cuenta con:

- variables de entrada con sus funciones de pertenencia;
- una variable de salida;
- una colección interpretable de reglas difusas.

El ciclo implementado es el clásico esquema Mamdani:

1. Fuzzificación
   Convierte cada entrada numérica en grados de pertenencia sobre sus
   etiquetas lingüísticas.
2. Evaluación de reglas
   Calcula la fuerza de activación de cada regla a partir de sus antecedentes.
3. Implicación
   Recorta el consecuente de cada regla según dicha fuerza.
4. Agregación
   Combina todos los consecuentes activados en una sola salida difusa.
5. Desfuzzificación
   Convierte la salida agregada en un valor crisp mediante centroide.

El objetivo de este diseño es mantener la inferencia separada de:

- la definición semántica del dispositivo;
- la simulación temporal;
- la interfaz gráfica;
- y la optimización genética.

Esa separación permite que el sistema sea defendible académicamente:
las reglas y las membresías representan el conocimiento; este motor lo
ejecuta de forma transparente y trazable.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from app.fuzzy.membership import FuzzyVariable, FuzzySet
from app.fuzzy.rules import FuzzyRule, RuleSet


class MamdaniInference:
    """
    Ejecuta inferencia Mamdani sobre un conjunto fijo de variables y reglas.

    Este objeto representa la etapa operacional del sistema difuso. En otras
    palabras, recibe entradas crisp ya normalizadas, consulta sus grados de
    pertenencia, evalúa reglas y devuelve una salida crisp.

    Convenciones implementadas:

    - conjunción `AND`: mínimo;
    - disyunción conceptual entre consecuentes: máximo;
    - implicación: mínimo entre la membresía del consecuente y la fuerza
      de disparo;
    - agregación: máximo punto a punto;
    - desfuzzificación: centroide.

    Estas elecciones corresponden a una implementación Mamdani clásica y
    priorizan interpretabilidad frente a sofisticación matemática extra.
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
        Ejecuta el pipeline completo de inferencia para un caso puntual.

        Este método es la entrada de más alto nivel del motor. Se usa cuando
        solo se necesita el valor final de salida y no una traza detallada.

        Flujo interno:

        1. calcula grados de pertenencia de las entradas;
        2. evalúa reglas y obtiene consecuentes activados;
        3. agrega la salida difusa;
        4. aplica centroide sobre la función agregada.

        Args:
            crisp_inputs: Diccionario con entradas numéricas ya expresadas en
                las variables del controlador. Ejemplo:
                `{'temp_error': 3.5, 'occupancy': 2.0, 'tariff': 0.8}`.

        Returns:
            Valor crisp final de la variable de salida.
        """
        membership_degrees = self._fuzzify(crisp_inputs)
        activated_outputs = self._evaluate_rules(membership_degrees)
        aggregated = self._aggregate(activated_outputs)
        crisp_output = self._defuzzify(aggregated)
        
        return crisp_output
    
    def _fuzzify(self, crisp_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Convierte entradas crisp en grados de pertenencia.

        Para cada variable de entrada:

        - busca el valor crisp correspondiente;
        - evalúa todas sus funciones de pertenencia;
        - y devuelve un diccionario con los grados por etiqueta.

        Si una variable no aparece en `crisp_inputs`, se asume que no aporta
        evidencia y por tanto todos sus grados se fijan en `0.0`.

        Returns:
            Estructura del tipo:
            `{nombre_variable: {nombre_conjunto: grado_pertenencia}}`
        """
        memberships = {}
        for var_name, variable in self.input_variables.items():
            if var_name in crisp_inputs:
                value = crisp_inputs[var_name]
                memberships[var_name] = variable.fuzzify(value)
            else:
                # Ausencia de la entrada: la variable no activa ningún conjunto.
                memberships[var_name] = {s: 0.0 for s in variable.sets}
        return memberships
    
    def _evaluate_rules(self, 
                        memberships: Dict[str, Dict[str, float]]
                        ) -> List[Tuple[str, float]]:
        """
        Evalúa todas las reglas y devuelve sus consecuentes activados.

        Cada regla produce, como máximo, una activación de la forma:

        - conjunto de salida activado;
        - fuerza de disparo asociada.

        La fuerza se calcula como:

        - mínimo de los antecedentes;
        - multiplicado por el peso de la regla.

        Solo se conservan reglas con activación positiva efectiva. Esto evita
        ensuciar la agregación con consecuentes que no aportan nada.

        Returns:
            Lista de tuplas `(nombre_conjunto_salida, fuerza_de_disparo)`.
        """
        activated = []
        
        for rule in self.rule_base.get_rules():
            firing_strength = self._compute_firing_strength(rule, memberships)
            
            firing_strength *= rule.weight
            
            if firing_strength > 1e-10:
                output_set_name = rule.consequent[1]
                activated.append((output_set_name, firing_strength))
        
        return activated
    
    def _compute_firing_strength(self, 
                                  rule: FuzzyRule,
                                  memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Calcula la fuerza de disparo de una regla individual.

        La fuerza de disparo resume qué tan compatible es el caso evaluado con
        los antecedentes de la regla.

        En esta implementación:

        - cada antecedente aporta un grado de pertenencia;
        - la conjunción `AND` se resuelve con el mínimo;
        - si falta una variable o conjunto, la regla se considera inactiva.

        Args:
            rule: Regla difusa a evaluar.
            memberships: Grados de pertenencia ya fuzzificados.

        Returns:
            Valor en `[0, 1]` antes de aplicar el peso de la regla.
        """
        strengths = []
        
        for var_name, set_name in rule.antecedents:
            if var_name in memberships and set_name in memberships[var_name]:
                strengths.append(memberships[var_name][set_name])
            else:
                return 0.0
        
        if not strengths:
            return 0.0
        
        return min(strengths)
    
    def _aggregate(self, 
                   activated_outputs: List[Tuple[str, float]]) -> np.ndarray:
        """
        Construye la salida difusa agregada del sistema.

        Cada regla activada aporta un consecuente recortado. La agregación toma
        el máximo punto a punto entre todos esos consecuentes sobre el universo
        de salida.

        Intuición:

        - cada regla “empuja” parcialmente hacia una etiqueta de salida;
        - la salida agregada reúne todos esos empujes;
        - el resultado es una sola función difusa que resume la decisión global.

        Args:
            activated_outputs: Lista de pares `(conjunto_salida, fuerza)`.

        Returns:
            Array con la función de pertenencia agregada evaluada en el
            universo de la variable de salida.
        """
        universe = self.output_variable.universe
        aggregated = np.zeros_like(universe)
        
        if not activated_outputs:
            return aggregated
        
        for set_name, strength in activated_outputs:
            if set_name in self.output_variable.sets:
                output_set = self.output_variable.sets[set_name]
                mf_values = output_set.evaluate(universe)
                clipped = np.minimum(mf_values, strength)
                aggregated = np.maximum(aggregated, clipped)
        
        return aggregated
    
    def _defuzzify(self, aggregated: np.ndarray) -> float:
        """
        Convierte la salida difusa agregada en un valor crisp por centroide.

        Definición teórica continua:

                        \int_{\Omega} x \,\mu_{agg}(x)\,dx
            y^* = ------------------------------------------
                        \int_{\Omega} \mu_{agg}(x)\,dx

        Aproximación numérica usada en esta implementación:

                        \sum_i x_i \,\mu_{agg}(x_i)
            y^* \approx -------------------------------
                            \sum_i \mu_{agg}(x_i)

        donde ``x_i`` son los puntos discretizados del universo de salida.

        Si el área agregada es prácticamente cero, no hay evidencia suficiente
        para favorecer una región concreta de la salida. En ese caso se usa
        el centro geométrico del universo como valor neutro por defecto.

        Args:
            aggregated: Función agregada de salida.

        Returns:
            Valor crisp final de la inferencia.
        """
        universe = self.output_variable.universe
        total_area = np.sum(aggregated)
        
        if total_area < 1e-10:
            return (self.output_variable.universe_range[0] + 
                    self.output_variable.universe_range[1]) / 2.0
        
        centroid = np.sum(universe * aggregated) / total_area
        return float(centroid)
    
    def get_rule_activations(self, 
                              crisp_inputs: Dict[str, float]
                              ) -> List[Tuple[FuzzyRule, float]]:
        """
        Retorna todas las reglas junto con su fuerza de activación.

        Este método existe para trazabilidad y análisis explicable. La GUI y la
        documentación del proyecto lo usan para mostrar:

        - qué reglas participaron en un caso;
        - cuáles dominaron la decisión;
        - y con qué intensidad se activó cada una.

        Args:
            crisp_inputs: Caso puntual a evaluar.

        Returns:
            Lista de tuplas `(regla, fuerza)` para toda la base de reglas,
            incluyendo reglas con activación nula.
        """
        memberships = self._fuzzify(crisp_inputs)
        results = []
        
        for rule in self.rule_base.get_rules():
            strength = self._compute_firing_strength(rule, memberships)
            strength *= rule.weight
            results.append((rule, strength))
        
        return results
