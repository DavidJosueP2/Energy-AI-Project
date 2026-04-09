"""
Evaluacion del algoritmo genetico con enfasis en ahorro
manteniendo el confort del controlador difuso base.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController
from app.genetic.chromosome import ChromosomeEncoder
from app.simulation.metrics import PerformanceMetrics, calculate_metrics
from app.simulation.simulator import Simulator


@dataclass
class OptimizationReference:
    """Metricas de referencia del controlador difuso base."""

    metrics: PerformanceMetrics
    comfort_floor: float


@dataclass
class CandidateRecord:
    """Candidato evaluado por el GA con sus metricas agregadas."""

    chromosome: np.ndarray
    optimization_score: float
    metrics: PerformanceMetrics
    feasible: bool
    rank_tuple: tuple


class FitnessEvaluator:
    """
    Evalua cromosomas buscando ahorro energetico y economico
    sin degradar el confort del controlador difuso base.
    """

    def __init__(self, config: AppConfig, base_controller: FuzzyController):
        self.config = config
        self.base_controller = base_controller
        self.encoder = ChromosomeEncoder(controller=base_controller)
        self._eval_count = 0
        self.reference = self._build_reference()
        self.best_candidate: Optional[CandidateRecord] = None

    def _build_reference(self) -> OptimizationReference:
        simulator = Simulator(self.config)
        result = simulator.run(self.base_controller.get_controller_function(), label="ga_reference")
        metrics = calculate_metrics(result.data, self.config.simulation, self.config.metrics)
        comfort_floor = max(metrics.comfort_percentage - 0.5, 0.0)
        return OptimizationReference(metrics=metrics, comfort_floor=comfort_floor)

    def evaluate(self, chromosome: np.ndarray) -> float:
        self._eval_count += 1
        try:
            controller = self._decode_controller(chromosome)
            result = Simulator(self.config).run(controller.get_controller_function(), label="ga_eval")
            metrics = calculate_metrics(result.data, self.config.simulation, self.config.metrics)
            score = self._optimization_score(metrics)
            self._register_candidate(chromosome, metrics, score)
            return score
        except Exception:
            return -5.0

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        fitnesses = np.zeros(len(population))
        for idx, chromosome in enumerate(population):
            fitnesses[idx] = self.evaluate(chromosome)
        return fitnesses

    def _decode_controller(self, chromosome: np.ndarray) -> FuzzyController:
        params = self.encoder.decode(chromosome)
        controller = self.base_controller.clone()
        controller.set_membership_params(params)
        return controller

    def _optimization_score(self, metrics: PerformanceMetrics) -> float:
        ref = self.reference.metrics

        energy_saving = self._relative_improvement(ref.total_energy_kwh, metrics.total_energy_kwh)
        cost_saving = self._relative_improvement(ref.total_cost, metrics.total_cost)
        peak_saving = self._relative_improvement(ref.peak_demand_kw, metrics.peak_demand_kw)
        variability_saving = self._relative_improvement(ref.control_variability, metrics.control_variability)

        comfort_delta = metrics.comfort_percentage - ref.comfort_percentage
        comfort_drop = max(self.reference.comfort_floor - metrics.comfort_percentage, 0.0)
        avg_dev_worsening = max(
            (metrics.avg_temp_deviation - ref.avg_temp_deviation) / max(ref.avg_temp_deviation, 0.1),
            0.0,
        )
        max_dev_worsening = max(
            (metrics.max_temp_deviation - ref.max_temp_deviation) / max(ref.max_temp_deviation, 0.1),
            0.0,
        )

        # Penaliza con fuerza perder confort respecto al controlador base.
        comfort_penalty = (comfort_drop / 100.0) * 5.0

        savings_core = 0.65 * cost_saving + 0.25 * energy_saving
        operational_bonus = 0.0
        if cost_saving > 0.0 or energy_saving > 0.0:
            operational_bonus = 0.06 * peak_saving + 0.04 * variability_saving

        # Mejora de confort muy levemente premiada solo si no empeora ahorro.
        comfort_bonus = 0.0
        if savings_core >= 0.0:
            comfort_bonus = min(max(comfort_delta, 0.0), 1.0) / 500.0

        score = (
            savings_core
            + operational_bonus
            + comfort_bonus
            - comfort_penalty
            - 0.08 * avg_dev_worsening
            - 0.06 * max_dev_worsening
        )
        return float(score)

    def _register_candidate(self, chromosome: np.ndarray, metrics: PerformanceMetrics, score: float):
        ref = self.reference.metrics
        feasible = (
            metrics.comfort_percentage >= ref.comfort_percentage
            and metrics.total_cost <= ref.total_cost
            and metrics.total_energy_kwh <= ref.total_energy_kwh
        )
        comfort_gain = metrics.comfort_percentage - ref.comfort_percentage
        cost_saving = self._relative_improvement(ref.total_cost, metrics.total_cost)
        energy_saving = self._relative_improvement(ref.total_energy_kwh, metrics.total_energy_kwh)

        if feasible:
            rank_tuple = (
                1,
                round(cost_saving + energy_saving, 8),
                round(comfort_gain / 100.0, 8),
                round(score, 8),
            )
        else:
            rank_tuple = (
                0,
                round(score, 8),
                round(-max(ref.total_cost - metrics.total_cost, 0.0), 8),
                round(-max(ref.total_energy_kwh - metrics.total_energy_kwh, 0.0), 8),
            )

        candidate = CandidateRecord(
            chromosome=chromosome.copy(),
            optimization_score=float(score),
            metrics=metrics,
            feasible=feasible,
            rank_tuple=rank_tuple,
        )

        if self.best_candidate is None or candidate.rank_tuple > self.best_candidate.rank_tuple:
            self.best_candidate = candidate

    @staticmethod
    def _relative_improvement(reference_value: float, candidate_value: float) -> float:
        if abs(reference_value) < 1e-9:
            return 0.0
        return (reference_value - candidate_value) / abs(reference_value)

    @property
    def evaluations_count(self) -> int:
        return self._eval_count

    def reset_counter(self):
        self._eval_count = 0

    def get_best_candidate(self) -> Optional[CandidateRecord]:
        return self.best_candidate
