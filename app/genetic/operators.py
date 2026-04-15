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
    
    Selecciona un individuo mediante un torneo entre candidatos aleatorios.

    El operador:

    1. elige varios índices al azar;
    2. compara sus fitness;
    3. devuelve una copia del cromosoma ganador.

    Este mecanismo aumenta la probabilidad de seleccionar individuos buenos
    sin volver la selección completamente determinista.
    
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
    Cruce BLX-α (Blend Crossover) para genes reales.

    Para cada gen se observa el intervalo definido por ambos padres, se lo
    expande con el parámetro `alpha` y luego se muestrean los hijos con una
    distribución uniforme dentro de ese rango ampliado.

    Intuición:

    - si `alpha = 0`, los hijos quedan entre los padres;
    - si `alpha > 0`, los hijos también pueden explorar ligeramente fuera de
      ese intervalo.

    Eso vuelve al operador útil para funciones de pertenencia, donde conviene
    explorar regiones cercanas del espacio continuo sin abandonar del todo la
    información heredada.
    
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

    Para cada gen:

    - decide aleatoriamente si se muta;
    - calcula el rango permitido del gen;
    - escala `sigma` según ese rango;
    - suma ruido gaussiano centrado en cero.

    El escalado por rango es importante porque en este proyecto no todos los
    genes viven en la misma escala numérica. Así se evita mutar con la misma
    intensidad un parámetro de tarifa en `[0,1]` y uno de salida en `[0,100]`.
    
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
    Preserva élites de la generación anterior dentro de la nueva generación.

    El procedimiento es:

    1. localizar los `elite_count` mejores fitness de la población anterior;
    2. localizar los `elite_count` peores fitness de la nueva población;
    3. reemplazar a los peores nuevos por las élites antiguas.

    Esto ayuda a que el algoritmo no pierda soluciones valiosas por efectos
    aleatorios del cruce y la mutación.
    
    Args:
        old_population, old_fitnesses: Población y fitness anteriores.
        new_population, new_fitnesses: Población y fitness nuevos.
        elite_count: Número de élites a preservar.
        
    Returns:
        Tupla (población_con_élites, fitness_con_élites).
    """
    if elite_count <= 0:
        return new_population, new_fitnesses
    
    elite_indices = np.argsort(old_fitnesses)[-elite_count:]
    
    worst_indices = np.argsort(new_fitnesses)[:elite_count]
    
    result_pop = new_population.copy()
    result_fit = new_fitnesses.copy()
    
    for i, (elite_idx, worst_idx) in enumerate(zip(elite_indices, worst_indices)):
        result_pop[worst_idx] = old_population[elite_idx].copy()
        result_fit[worst_idx] = old_fitnesses[elite_idx]
    
    return result_pop, result_fit
