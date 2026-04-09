# ==============================================================================
# chromosome.py - Codificación de cromosomas para el algoritmo genético
# ==============================================================================
"""
Define la estructura del cromosoma que codifica los parámetros del 
controlador difuso para su optimización mediante algoritmo genético.

El cromosoma es un vector de valores reales que codifica:
- Parámetros de funciones de pertenencia de variables de entrada
- Parámetros de funciones de pertenencia de la variable de salida

Cada función de pertenencia triangular se codifica con 3 parámetros [a, b, c].
El cromosoma total agrupa todos estos parámetros en un vector plano.

Restricciones importantes:
- a <= b <= c para cada función triangular
- Los conjuntos de una variable deben mantener orden creciente de centros
- Los parámetros deben estar dentro de los rangos del universo de discurso
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from app.config import FuzzyConfig


class ChromosomeEncoder:
    """
    Codifica y decodifica los parámetros del controlador difuso
    en un vector cromosómico para el algoritmo genético.
    """
    
    def __init__(self, fuzzy_config: FuzzyConfig):
        """
        Args:
            fuzzy_config: Configuración del controlador difuso con
                         parámetros iniciales de funciones de pertenencia.
        """
        self.config = fuzzy_config
        
        # Definir la estructura del cromosoma: qué parámetros se optimizan
        # y sus rangos válidos
        self._variable_specs = self._build_variable_specs()
        self._gene_specs = self._build_gene_specs()
        self.chromosome_length = len(self._gene_specs)
    
    def _build_variable_specs(self) -> List[Tuple[str, str, Dict[str, List[float]], Tuple[float, float]]]:
        """
        Construye la especificación de variables y sus conjuntos.
        
        Returns:
            Lista de (nombre_variable, tipo, sets_params, universe_range)
        """
        cfg = self.config
        specs = [
            ('temp_error', 'input', cfg.temp_error_sets, cfg.temp_error_range),
            ('humidity', 'input', cfg.humidity_sets, cfg.humidity_range),
            ('occupancy', 'input', cfg.occupancy_sets, cfg.occupancy_range),
            ('tariff', 'input', cfg.tariff_sets, cfg.tariff_range),
            ('consumption', 'input', cfg.consumption_sets, cfg.consumption_range),
            ('hvac_output', 'output', cfg.output_sets, cfg.output_range),
        ]
        return specs
    
    def _build_gene_specs(self) -> List[Dict]:
        """
        Construye la especificación de cada gen del cromosoma.
        Cada gen tiene: índice, variable asociada, conjunto, parámetro (a/b/c),
        valor por defecto y rango válido.
        """
        genes = []
        
        for var_name, var_type, sets_params, universe_range in self._variable_specs:
            u_min, u_max = universe_range
            margin = (u_max - u_min) * 0.15  # Margen del 15% fuera del rango
            
            for set_name, params in sets_params.items():
                for p_idx, p_val in enumerate(params):
                    param_label = ['a', 'b', 'c'][p_idx] if len(params) == 3 else f'p{p_idx}'
                    genes.append({
                        'variable': var_name,
                        'set': set_name,
                        'param': param_label,
                        'param_idx': p_idx,
                        'default': p_val,
                        'min': u_min - margin,
                        'max': u_max + margin,
                    })
        
        return genes
    
    def encode_default(self) -> np.ndarray:
        """
        Codifica los parámetros por defecto en un vector cromosómico.
        
        Returns:
            Array numpy con los valores por defecto.
        """
        return np.array([g['default'] for g in self._gene_specs])
    
    def decode(self, chromosome: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        """
        Decodifica un vector cromosómico a parámetros de funciones de pertenencia.
        
        Args:
            chromosome: Vector de valores reales.
            
        Returns:
            Dict {variable: {conjunto: [params]}} listo para FuzzyController.
        """
        params = {}
        
        for i, gene in enumerate(self._gene_specs):
            var_name = gene['variable']
            set_name = gene['set']
            p_idx = gene['param_idx']
            
            if var_name not in params:
                params[var_name] = {}
            if set_name not in params[var_name]:
                params[var_name][set_name] = [0.0, 0.0, 0.0]
            
            params[var_name][set_name][p_idx] = chromosome[i]
        
        return params
    
    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Repara un cromosoma para garantizar restricciones válidas:
        1. Cada gen dentro de su rango
        2. Para cada función triangular: a <= b <= c
        3. Parámetros dentro del universo de discurso (con margen)
        
        Args:
            chromosome: Vector a reparar.
            
        Returns:
            Vector reparado.
        """
        repaired = chromosome.copy()
        
        # 1. Clamp cada gen a su rango
        for i, gene in enumerate(self._gene_specs):
            repaired[i] = np.clip(repaired[i], gene['min'], gene['max'])
        
        # 2. Garantizar a <= b <= c para cada función triangular
        decoded = self.decode(repaired)
        
        for var_name, sets_dict in decoded.items():
            for set_name, params in sets_dict.items():
                if len(params) == 3:
                    a, b, c = params
                    # Ordenar
                    a, b, c = sorted([a, b, c])
                    # Asegurar separación mínima
                    min_sep = 0.1
                    if b - a < min_sep:
                        b = a + min_sep
                    if c - b < min_sep:
                        c = b + min_sep
                    params[0], params[1], params[2] = a, b, c
        
        # Re-codificar el cromosoma reparado
        repaired = self._encode_from_decoded(decoded)
        
        return repaired
    
    def _encode_from_decoded(self, decoded: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
        """Codifica parámetros decodificados de vuelta a un vector."""
        chromosome = np.zeros(self.chromosome_length)
        
        for i, gene in enumerate(self._gene_specs):
            var_name = gene['variable']
            set_name = gene['set']
            p_idx = gene['param_idx']
            
            if var_name in decoded and set_name in decoded[var_name]:
                chromosome[i] = decoded[var_name][set_name][p_idx]
            else:
                chromosome[i] = gene['default']
        
        return chromosome
    
    def get_gene_info(self) -> List[Dict]:
        """Retorna información de todos los genes para análisis."""
        return deepcopy(self._gene_specs)
    
    def generate_random(self, rng: np.random.RandomState, 
                        perturbation: float = 2.0) -> np.ndarray:
        """
        Genera un cromosoma aleatorio perturbando los valores por defecto.
        
        Args:
            rng: Generador de números aleatorios.
            perturbation: Máxima perturbación respecto al valor por defecto.
            
        Returns:
            Cromosoma aleatorio válido.
        """
        default = self.encode_default()
        noise = rng.uniform(-perturbation, perturbation, self.chromosome_length)
        
        # Escalar la perturbación según el rango de cada gen
        for i, gene in enumerate(self._gene_specs):
            gene_range = gene['max'] - gene['min']
            noise[i] *= (gene_range / 20.0)  # Perturbación proporcional
        
        chromosome = default + noise
        return self.repair(chromosome)
