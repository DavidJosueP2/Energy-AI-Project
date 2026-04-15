import numpy as np
from typing import Dict, List, Tuple, Optional

from app.fuzzy.membership import FuzzyVariable, FuzzySet
from app.fuzzy.rules import FuzzyRule, RuleSet


class MamdaniInference:
    """
    Convenciones implementadas:

    - conjunción `AND`: mínimo;
    - disyunción conceptual entre consecuentes: máximo;
    - implicación: mínimo entre la membresía del consecuente y la fuerza
      de disparo;
    - agregación: máximo punto a punto;
    - desfuzzificación: centroide.
    """
    
    def __init__(self, 
                 input_variables: Dict[str, FuzzyVariable],
                 output_variable: FuzzyVariable,
                 fuzzy_rule_set: RuleSet):
        """
        Args:
            input_variables: Diccionario {nombre: FuzzyVariable} de entradas.
            output_variable: Variable difusa de salida.
            rule_base: Base de reglas a aplicar.
        """
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.rule_base = fuzzy_rule_set
    
    def infer(self, crisp_inputs: Dict[str, float]) -> float:
        membership_degrees = self._fuzzify(crisp_inputs)
        activated_outputs = self._evaluate_rules(membership_degrees)
        implied_outputs = self._implicate(activated_outputs)
        aggregated = self._aggregate(implied_outputs)
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
    
    def _implicate(
        self,
        activated_outputs: List[Tuple[str, float]],
    ) -> List[np.ndarray]:
        """
        Materializa los consecuentes difusos de cada regla activada.

        1. tomar la membresía del conjunto de salida activado;
        2. recortarla con la fuerza de disparo de la regla;
        3. devolver la colección de salidas implicadas.

        Matemáticamente, para cada regla activada con consecuente ``B_r`` y
        fuerza ``alpha_r``:

            mu'_r(y) = min(mu_{B_r}(y), alpha_r)

        Args:
            activated_outputs: Lista de pares `(conjunto_salida, fuerza)`.

        Returns:
            Lista de arrays, uno por consecuente ya implicado.
        """
        universe = self.output_variable.universe
        implied_outputs: List[np.ndarray] = []

        for set_name, strength in activated_outputs:
            if set_name in self.output_variable.sets:
                output_set = self.output_variable.sets[set_name]
                mf_values = output_set.evaluate(universe)
                implied_outputs.append(np.minimum(mf_values, strength))

        return implied_outputs

    def _aggregate(
        self,
        implied_outputs: List[np.ndarray],
    ) -> np.ndarray:
        """
        Construye la salida difusa agregada del sistema.

        Una vez implicados los consecuentes de cada regla, la agregación toma
        el máximo punto a punto entre todas esas salidas parciales sobre el
        universo de salida.

        Intuición:

        - cada regla ya dejó una contribución difusa recortada;
        - la agregación reúne todas esas contribuciones;
        - el resultado es una sola función difusa que resume la decisión global.

        Args:
            implied_outputs: Lista de arrays con los consecuentes ya
                implicados de cada regla.

        Returns:
            Array con la función de pertenencia agregada evaluada en el
            universo de la variable de salida.
        """
        universe = self.output_variable.universe
        aggregated = np.zeros_like(universe)
        
        if not implied_outputs:
            return aggregated
        
        for implied in implied_outputs:
            aggregated = np.maximum(aggregated, implied)
        
        return aggregated
    
    def _defuzzify(self, aggregated: np.ndarray) -> float:
        r"""
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
