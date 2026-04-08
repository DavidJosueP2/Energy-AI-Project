# ==============================================================================
# fitness.py - Función de aptitud para el algoritmo genético
# ==============================================================================
"""
Evalúa la calidad de un cromosoma ejecutando la simulación completa
con el controlador difuso parametrizado y calculando un score multiobjetivo.
"""

import numpy as np
from typing import Optional, Callable

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController
from app.genetic.chromosome import ChromosomeEncoder
from app.simulation.simulator import Simulator
from app.simulation.metrics import calculate_fitness


def evaluate_chromosome(chromosome: np.ndarray,
                        encoder: ChromosomeEncoder,
                        base_controller: FuzzyController,
                        config: AppConfig) -> float:
    """
    Evalúa un cromosoma individual.
    
    Proceso:
    1. Decodificar cromosoma → parámetros de funciones de pertenencia
    2. Crear controlador con esos parámetros
    3. Ejecutar simulación completa
    4. Calcular fitness multiobjetivo
    
    Args:
        chromosome: Vector cromosómico a evaluar.
        encoder: Codificador para decodificar el cromosoma.
        base_controller: Controlador base a clonar y modificar.
        config: Configuración global.
        
    Returns:
        Score de fitness (mayor = mejor).
    """
    try:
        # 1. Decodificar
        params = encoder.decode(chromosome)
        
        # 2. Crear controlador modificado
        controller = base_controller.clone()
        controller.set_membership_params(params)
        
        # 3. Ejecutar simulación
        simulator = Simulator(config)
        result = simulator.run(controller.get_controller_function(), label="ga_eval")
        
        # 4. Calcular fitness
        fitness = calculate_fitness(
            result.data, 
            config.simulation, 
            config.metrics
        )
        
        return float(fitness)
        
    except Exception as e:
        # Cromosoma inválido → fitness muy bajo
        return -1.0


class FitnessEvaluator:
    """
    Evaluador de fitness que encapsula la configuración y permite
    evaluar múltiples cromosomas de forma eficiente.
    """
    
    def __init__(self, config: AppConfig, base_controller: FuzzyController):
        self.config = config
        self.base_controller = base_controller
        self.encoder = ChromosomeEncoder(config.fuzzy)
        self._eval_count = 0
    
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evalúa un cromosoma y retorna su fitness."""
        self._eval_count += 1
        return evaluate_chromosome(
            chromosome, self.encoder, self.base_controller, self.config
        )
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evalúa toda una población de cromosomas.
        
        Args:
            population: Matriz (n_individuos × n_genes).
            
        Returns:
            Array de fitness para cada individuo.
        """
        fitnesses = np.zeros(len(population))
        for i, chromosome in enumerate(population):
            fitnesses[i] = self.evaluate(chromosome)
        return fitnesses
    
    @property
    def evaluations_count(self) -> int:
        """Número total de evaluaciones realizadas."""
        return self._eval_count
    
    def reset_counter(self):
        """Reinicia el contador de evaluaciones."""
        self._eval_count = 0
