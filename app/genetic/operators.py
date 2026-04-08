# ==============================================================================
# operators.py - Operadores genéticos: selección, cruce, mutación
# ==============================================================================
"""
Implementa los operadores genéticos fundamentales:
- Selección por torneo
- Cruce BLX-α (Blend Crossover)
- Mutación gaussiana
- Elitismo

Estos operadores trabajan sobre vectores de valores reales (cromosomas)
que codifican parámetros de funciones de pertenencia difusa.
"""

import numpy as np
from typing import Tuple

from app.config import GeneticConfig
from app.genetic.chromosome import ChromosomeEncoder


def tournament_selection(population: np.ndarray,
                         fitnesses: np.ndarray,
                         tournament_size: int,
                         rng: np.random.RandomState) -> np.ndarray:
    """
    Selección por torneo.
    
    Selecciona un individuo eligiendo aleatoriamente `tournament_size`
    candidatos y retornando el mejor (mayor fitness).
    
    Args:
        population: Matriz de población (n × genes).
        fitnesses: Vector de fitness por individuo.
        tournament_size: Número de candidatos por torneo.
        rng: Generador aleatorio.
        
    Returns:
        Cromosoma del ganador del torneo.
    """
    n = len(population)
    candidates = rng.choice(n, size=min(tournament_size, n), replace=False)
    winner = candidates[np.argmax(fitnesses[candidates])]
    return population[winner].copy()


def blx_alpha_crossover(parent1: np.ndarray,
                         parent2: np.ndarray,
                         alpha: float,
                         rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cruce BLX-α (Blend Crossover).
    
    Para cada gen i:
    - d = |parent1[i] - parent2[i]|
    - min_val = min(parent1[i], parent2[i]) - α*d
    - max_val = max(parent1[i], parent2[i]) + α*d
    - hijo[i] = uniform(min_val, max_val)
    
    Este operador permite exploración más allá del rango de los padres
    (cuando α > 0), lo cual es beneficioso para optimización de parámetros
    continuos.
    
    Args:
        parent1, parent2: Cromosomas de los padres.
        alpha: Factor de extensión (típicamente 0.5).
        rng: Generador aleatorio.
        
    Returns:
        Tupla de dos hijos.
    """
    n_genes = len(parent1)
    child1 = np.zeros(n_genes)
    child2 = np.zeros(n_genes)
    
    for i in range(n_genes):
        p_min = min(parent1[i], parent2[i])
        p_max = max(parent1[i], parent2[i])
        d = p_max - p_min
        
        low = p_min - alpha * d
        high = p_max + alpha * d
        
        child1[i] = rng.uniform(low, high)
        child2[i] = rng.uniform(low, high)
    
    return child1, child2


def gaussian_mutation(chromosome: np.ndarray,
                      mutation_prob: float,
                      sigma: float,
                      encoder: ChromosomeEncoder,
                      rng: np.random.RandomState) -> np.ndarray:
    """
    Mutación gaussiana gen por gen.
    
    Para cada gen, con probabilidad `mutation_prob`, se añade
    ruido gaussiano N(0, σ) al valor del gen.
    
    Args:
        chromosome: Cromosoma a mutar.
        mutation_prob: Probabilidad de mutación por gen.
        sigma: Desviación estándar de la mutación.
        encoder: Codificador para reparar restricciones.
        rng: Generador aleatorio.
        
    Returns:
        Cromosoma mutado.
    """
    mutated = chromosome.copy()
    gene_specs = encoder.get_gene_info()
    
    for i in range(len(mutated)):
        if rng.random() < mutation_prob:
            # Escalar sigma según el rango del gen
            gene_range = gene_specs[i]['max'] - gene_specs[i]['min']
            scaled_sigma = sigma * (gene_range / 20.0)
            mutated[i] += rng.normal(0, scaled_sigma)
    
    return mutated


def apply_elitism(old_population: np.ndarray,
                  old_fitnesses: np.ndarray,
                  new_population: np.ndarray,
                  new_fitnesses: np.ndarray,
                  elite_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica elitismo: preserva los mejores individuos de la generación anterior.
    
    Reemplaza los peores individuos de la nueva generación con los
    mejores de la anterior.
    
    Args:
        old_population, old_fitnesses: Población y fitness anteriores.
        new_population, new_fitnesses: Población y fitness nuevos.
        elite_count: Número de élites a preservar.
        
    Returns:
        Tupla (población_con_élites, fitness_con_élites).
    """
    if elite_count <= 0:
        return new_population, new_fitnesses
    
    # Encontrar los mejores de la generación anterior
    elite_indices = np.argsort(old_fitnesses)[-elite_count:]
    
    # Encontrar los peores de la nueva generación
    worst_indices = np.argsort(new_fitnesses)[:elite_count]
    
    # Reemplazar
    result_pop = new_population.copy()
    result_fit = new_fitnesses.copy()
    
    for i, (elite_idx, worst_idx) in enumerate(zip(elite_indices, worst_indices)):
        result_pop[worst_idx] = old_population[elite_idx].copy()
        result_fit[worst_idx] = old_fitnesses[elite_idx]
    
    return result_pop, result_fit
