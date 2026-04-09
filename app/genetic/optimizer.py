# ==============================================================================
# optimizer.py - Optimizador principal del algoritmo genético
# ==============================================================================
"""
Implementa el bucle principal del algoritmo genético para optimizar
los parámetros del controlador difuso.

Flujo del GA:
1. Generar población inicial (perturbada desde parámetros por defecto)
2. Evaluar fitness de cada individuo
3. Repetir por N generaciones:
   a. Seleccionar padres (torneo)
   b. Cruzar (BLX-α)
   c. Mutar (gaussiana)
   d. Reparar restricciones
   e. Evaluar nueva generación
   f. Aplicar elitismo
   g. Registrar estadísticas
4. Retornar el mejor individuo encontrado
"""

import numpy as np
import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, field

from app.config import AppConfig, GeneticConfig
from app.fuzzy.controller import FuzzyController
from app.genetic.chromosome import ChromosomeEncoder
from app.genetic.fitness import FitnessEvaluator
from app.genetic.operators import (
    tournament_selection,
    blx_alpha_crossover,
    gaussian_mutation,
    apply_elitism,
)


@dataclass
class GenerationStats:
    """Estadísticas de una generación del GA."""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    std_fitness: float = 0.0
    best_chromosome: Optional[np.ndarray] = None
    elapsed_seconds: float = 0.0


@dataclass
class OptimizationResult:
    """Resultado completo de la optimización genética."""
    best_chromosome: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = 0.0
    history: List[GenerationStats] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_seconds: float = 0.0
    converged: bool = False
    
    def get_fitness_history(self) -> List[float]:
        """Retorna la evolución del mejor fitness por generación."""
        return [s.best_fitness for s in self.history]
    
    def get_avg_fitness_history(self) -> List[float]:
        """Retorna la evolución del fitness promedio por generación."""
        return [s.avg_fitness for s in self.history]


class GeneticOptimizer:
    """
    Algoritmo genético para optimización del controlador difuso.
    """
    
    def __init__(self, config: AppConfig, base_controller: FuzzyController):
        """
        Args:
            config: Configuración global (incluye GeneticConfig).
            base_controller: Controlador difuso base a optimizar.
        """
        self.config = config
        self.ga_config = config.genetic
        self.base_controller = base_controller
        
        # Componentes
        self.encoder = ChromosomeEncoder(controller=base_controller)
        self.evaluator = FitnessEvaluator(config, base_controller)
        
        # Estado
        self.rng = np.random.RandomState(self.ga_config.random_seed)
        self._is_running = False
        self._should_stop = False
    
    def optimize(self,
                 progress_callback: Optional[Callable[[int, int, float], None]] = None
                 ) -> OptimizationResult:
        """
        Ejecuta la optimización genética completa.
        
        Args:
            progress_callback: Función(generacion, total, mejor_fitness) para progreso.
            
        Returns:
            OptimizationResult con el mejor cromosoma y estadísticas.
        """
        ga = self.ga_config
        self._is_running = True
        self._should_stop = False
        start_time = time.time()
        
        result = OptimizationResult()
        
        # 1. Generar población inicial
        population = self._init_population()
        
        # 2. Evaluar fitness inicial
        fitnesses = self.evaluator.evaluate_population(population)
        
        # Registrar generación 0
        stats = self._compute_stats(0, population, fitnesses, start_time)
        result.history.append(stats)
        
        if progress_callback:
            progress_callback(0, ga.num_generations, stats.best_fitness)
        
        # 3. Bucle generacional
        for gen in range(1, ga.num_generations + 1):
            if self._should_stop:
                break
            
            # a-d. Crear nueva generación
            new_population = self._create_next_generation(population, fitnesses)
            
            # e. Evaluar nueva generación
            new_fitnesses = self.evaluator.evaluate_population(new_population)
            
            # f. Aplicar elitismo
            population, fitnesses = apply_elitism(
                population, fitnesses,
                new_population, new_fitnesses,
                ga.elitism_count
            )
            
            # g. Registrar estadísticas
            stats = self._compute_stats(gen, population, fitnesses, start_time)
            result.history.append(stats)
            
            if progress_callback:
                progress_callback(gen, ga.num_generations, stats.best_fitness)
        
        # 4. Compilar resultado final
        best_candidate = self.evaluator.get_best_candidate()
        if best_candidate is not None:
            result.best_chromosome = best_candidate.chromosome.copy()
            result.best_fitness = best_candidate.optimization_score
        else:
            best_idx = np.argmax(fitnesses)
            result.best_chromosome = population[best_idx].copy()
            result.best_fitness = fitnesses[best_idx]
        result.total_evaluations = self.evaluator.evaluations_count
        result.total_time_seconds = time.time() - start_time
        
        self._is_running = False
        return result
    
    def stop(self):
        """Detiene la optimización prematuramente."""
        self._should_stop = True
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    def _init_population(self) -> np.ndarray:
        """Genera la población inicial."""
        ga = self.ga_config
        pop_size = ga.population_size
        n_genes = self.encoder.chromosome_length
        
        population = np.zeros((pop_size, n_genes))
        
        # Primer individuo: parámetros por defecto (asegurar baseline)
        population[0] = self.encoder.encode_default()
        
        # Resto: perturbaciones aleatorias del default
        for i in range(1, pop_size):
            population[i] = self.encoder.generate_random(
                self.rng, ga.init_perturbation
            )
        
        return population
    
    def _create_next_generation(self, 
                                 population: np.ndarray,
                                 fitnesses: np.ndarray) -> np.ndarray:
        """Crea la siguiente generación mediante selección, cruce y mutación."""
        ga = self.ga_config
        pop_size = len(population)
        n_genes = self.encoder.chromosome_length
        new_pop = np.zeros((pop_size, n_genes))
        
        i = 0
        while i < pop_size:
            # Selección
            parent1 = tournament_selection(
                population, fitnesses, ga.tournament_size, self.rng
            )
            parent2 = tournament_selection(
                population, fitnesses, ga.tournament_size, self.rng
            )
            
            # Cruce
            if self.rng.random() < ga.crossover_probability:
                child1, child2 = blx_alpha_crossover(
                    parent1, parent2, ga.blx_alpha, self.rng
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutación
            child1 = gaussian_mutation(
                child1, ga.mutation_probability, ga.mutation_sigma,
                self.encoder, self.rng
            )
            child2 = gaussian_mutation(
                child2, ga.mutation_probability, ga.mutation_sigma,
                self.encoder, self.rng
            )
            
            # Reparar restricciones
            child1 = self.encoder.repair(child1)
            child2 = self.encoder.repair(child2)
            
            # Insertar en nueva población
            new_pop[i] = child1
            i += 1
            if i < pop_size:
                new_pop[i] = child2
                i += 1
        
        return new_pop
    
    def _compute_stats(self, generation: int, 
                        population: np.ndarray,
                        fitnesses: np.ndarray,
                        start_time: float) -> GenerationStats:
        """Calcula estadísticas de la generación actual."""
        best_idx = np.argmax(fitnesses)
        return GenerationStats(
            generation=generation,
            best_fitness=float(fitnesses[best_idx]),
            avg_fitness=float(np.mean(fitnesses)),
            worst_fitness=float(np.min(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            best_chromosome=population[best_idx].copy(),
            elapsed_seconds=time.time() - start_time,
        )
    
    def decode_best(self, result: OptimizationResult) -> FuzzyController:
        """
        Decodifica el mejor cromosoma y crea un controlador optimizado.
        
        Args:
            result: Resultado de la optimización.
            
        Returns:
            FuzzyController con parámetros optimizados.
        """
        params = self.encoder.decode(result.best_chromosome)
        controller = self.base_controller.clone()
        controller.set_membership_params(params)
        return controller
