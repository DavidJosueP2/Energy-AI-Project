# ==============================================================================
# test_genetic.py - Pruebas del módulo de algoritmo genético
# ==============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.config import AppConfig, GeneticConfig, FuzzyConfig
from app.fuzzy.controller import FuzzyController
from app.genetic.chromosome import ChromosomeEncoder
from app.genetic.operators import (
    tournament_selection, blx_alpha_crossover, gaussian_mutation
)
from app.genetic.fitness import FitnessEvaluator
from app.genetic.optimizer import GeneticOptimizer
from app.fuzzy.membership import triangular_mf, trapezoidal_mf


class TestChromosomeEncoder:
    """Pruebas para la codificación de cromosomas."""

    def setup_method(self):
        self.encoder = ChromosomeEncoder()

    def test_chromosome_length(self):
        """El cromosoma debe tener longitud > 0."""
        assert self.encoder.chromosome_length > 0

    def test_encode_default(self):
        """Codificar por defecto debe retornar un vector válido."""
        ch = self.encoder.encode_default()
        assert len(ch) == self.encoder.chromosome_length
        assert np.all(np.isfinite(ch))

    def test_decode_roundtrip(self):
        """Codificar y decodificar debe preservar la estructura."""
        ch = self.encoder.encode_default()
        decoded = self.encoder.decode(ch)
        assert 'temp_error' in decoded
        assert 'hvac_output' in decoded

    def test_repair_enforces_order(self):
        """Reparar debe garantizar parámetros no decrecientes."""
        ch = self.encoder.encode_default()
        # Desordenar algunos genes
        ch[0] = 10.0
        ch[1] = 5.0
        ch[2] = 7.0
        repaired = self.encoder.repair(ch)
        decoded = self.encoder.decode(repaired)
        
        for var_name, sets_dict in decoded.items():
            for set_name, params in sets_dict.items():
                assert all(params[idx] <= params[idx + 1] for idx in range(len(params) - 1)), \
                    f"{var_name}.{set_name}: {params} no cumple orden no decreciente"

    def test_repair_preserves_explicit_trapezoidal_shoulders(self):
        """Las familias extremas trapezoidales deben seguir siendo hombros tras repair()."""
        ch = self.encoder.encode_default()
        repaired = self.encoder.repair(ch)
        decoded = self.encoder.decode(repaired)

        left = decoded["temp_error"]["baja"]
        right = decoded["hvac_output"]["muy_alta"]

        assert len(left) == 4
        assert left[0] == left[1]
        assert left[2] > left[1]
        assert left[3] > left[2]

        assert len(right) == 4
        assert right[2] == right[3]
        assert right[1] < right[2]
        assert right[0] < right[1]

    def test_random_generation(self):
        """Generar cromosoma aleatorio debe producir vector válido."""
        rng = np.random.RandomState(42)
        ch = self.encoder.generate_random(rng)
        assert len(ch) == self.encoder.chromosome_length
        assert np.all(np.isfinite(ch))

    def test_repair_preserves_membership_coverage(self):
        """Las funciones reparadas no deben dejar huecos sin pertenencia."""
        rng = np.random.RandomState(7)
        repaired = self.encoder.generate_random(rng)
        decoded = self.encoder.decode(repaired)

        for variable in self.encoder._variable_specs:
            universe = np.linspace(variable["range"][0], variable["range"][1], 400)[1:-1]
            total_membership = np.zeros_like(universe)

            for set_name, params in decoded[variable["name"]].items():
                if len(params) == 3:
                    membership = triangular_mf(universe, params)
                else:
                    membership = trapezoidal_mf(universe, params)
                total_membership = np.maximum(total_membership, membership)

            assert np.all(total_membership > 0.0), \
                f"La variable {variable['name']} quedo con huecos sin cobertura"


class TestGeneticOperators:
    """Pruebas para operadores genéticos."""

    def test_tournament_selection(self):
        """Selección por torneo debe retornar un individuo."""
        rng = np.random.RandomState(42)
        pop = rng.randn(10, 5)
        fit = rng.rand(10)
        selected = tournament_selection(pop, fit, 3, rng)
        assert len(selected) == 5

    def test_blx_crossover(self):
        """BLX-α debe producir dos hijos."""
        rng = np.random.RandomState(42)
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        c1, c2 = blx_alpha_crossover(p1, p2, 0.5, rng)
        assert len(c1) == 3
        assert len(c2) == 3
        # Los hijos no deben ser idénticos a los padres
        assert not np.array_equal(c1, p1)

    def test_mutation_changes_chromosome(self):
        """Mutación con alta probabilidad debe cambiar el cromosoma."""
        rng = np.random.RandomState(42)
        encoder = ChromosomeEncoder()
        ch = encoder.encode_default()
        mutated = gaussian_mutation(ch, 1.0, 2.0, encoder, rng)
        # Con prob=1.0, al menos algo debería cambiar
        assert not np.array_equal(ch, mutated)


class TestFitnessEvaluation:
    """Pruebas para la evaluación de fitness."""

    def test_default_fitness(self):
        """El cromosoma por defecto debe tener fitness razonable."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        controller = FuzzyController(config.fuzzy)
        evaluator = FitnessEvaluator(config, controller)
        
        ch = evaluator.encoder.encode_default()
        fitness = evaluator.evaluate(ch)
        assert isinstance(fitness, float)
        assert fitness > -2.0  # No debería ser terriblemente malo


class TestGeneticOptimizer:
    """Pruebas para el optimizador genético (ejecución corta)."""

    def test_short_optimization(self):
        """Una optimización muy corta debe completarse sin errores."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        config.genetic.population_size = 6
        config.genetic.num_generations = 2
        
        controller = FuzzyController(config.fuzzy)
        optimizer = GeneticOptimizer(config, controller)
        result = optimizer.optimize()
        
        assert result.best_fitness > -2.0
        assert len(result.history) > 0
        assert result.best_chromosome is not None
        assert result.total_evaluations > 0

    def test_decode_best(self):
        """Decodificar el mejor cromosoma debe producir un controlador funcional."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        config.genetic.population_size = 6
        config.genetic.num_generations = 2
        
        controller = FuzzyController(config.fuzzy)
        optimizer = GeneticOptimizer(config, controller)
        result = optimizer.optimize()
        
        opt_controller = optimizer.decode_best(result)
        output = opt_controller.evaluate({
            'temp_error': 5.0,
            'occupancy': 2.0,
            'tariff_normalized': 0.5,
            'consumption_normalized': 0.5,
        })
        assert 0 <= output <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
