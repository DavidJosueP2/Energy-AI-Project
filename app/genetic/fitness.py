"""
Evaluación de fitness para la optimización genética del controlador difuso.

Este módulo define cómo se juzga si un cromosoma representa una mejora real
respecto al controlador base. La idea central es importante:

- el GA no optimiza directamente temperaturas, costos o reglas aisladas;
- optimiza parámetros de membresía del sistema difuso;
- y cada candidato se evalúa ejecutando una simulación completa.

Por eso el fitness aquí no es una fórmula abstracta desconectada del sistema,
sin la traducción numérica de una comparación operativa entre:

- el controlador difuso base;
- y una variante difusa construida a partir de un cromosoma.

La política de evaluación prioriza:

- ahorro económico;
- ahorro energético;
- mantenimiento del confort;
- y rechazo de soluciones que, aunque tengan score aparente, sean poco
  defendibles frente al controlador base.
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
    Evalúa cromosomas del GA contra una referencia construida con el
    controlador difuso original.

    Cada cromosoma se interpreta como una parametrización alternativa de las
    funciones de pertenencia. El proceso de evaluación es:

    1. decodificar el cromosoma;
    2. reconstruir un controlador difuso candidato;
    3. correr una simulación temporal completa;
    4. calcular métricas;
    5. convertir esas métricas en un score escalar.

    Aunque el GA necesita una puntuación escalar para evolucionar, este
    evaluador intenta preservar la lógica multiobjetivo del problema:
    confort, costo, energía y comportamiento operativo.
    """

    def __init__(self, config: AppConfig, base_controller: FuzzyController):
        self.config = config
        self.base_controller = base_controller
        self.encoder = ChromosomeEncoder(controller=base_controller)
        self._eval_count = 0
        self.reference = self._build_reference()
        self.best_candidate: Optional[CandidateRecord] = None

    def _build_reference(self) -> OptimizationReference:
        """
        Construye la referencia base contra la cual se compararán los candidatos.

        Esta simulación se ejecuta una sola vez al crear el evaluador y sirve
        para fijar:

        - el nivel de confort del controlador original;
        - el costo original;
        - el consumo original;
        - y un piso de confort aceptable.

        Returns:
            Objeto con métricas base y umbral mínimo de confort admisible.
        """
        simulator = Simulator(self.config)
        result = simulator.run(self.base_controller.get_controller_function(), label="ga_reference")
        metrics = calculate_metrics(result.data, self.config.simulation, self.config.metrics)
        comfort_floor = max(metrics.comfort_percentage - 0.5, 0.0)
        return OptimizationReference(metrics=metrics, comfort_floor=comfort_floor)

    def evaluate(self, chromosome: np.ndarray) -> float:
        """
        Evalúa un cromosoma individual.

        El cromosoma no se evalúa directamente. Primero se convierte en un
        controlador difuso candidato y después se lo pone a prueba en la
        simulación completa del sistema.

        Args:
            chromosome: Vector real con parámetros de membresía codificados.

        Returns:
            Score escalar del candidato. Si ocurre un error estructural o de
            simulación, se retorna una penalización fuerte.
        """
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
        """
        Evalúa secuencialmente toda una población.

        Args:
            population: Matriz `n_individuos x n_genes`.

        Returns:
            Vector de fitness por individuo.
        """
        fitnesses = np.zeros(len(population))
        for idx, chromosome in enumerate(population):
            fitnesses[idx] = self.evaluate(chromosome)
        return fitnesses

    def _decode_controller(self, chromosome: np.ndarray) -> FuzzyController:
        """
        Reconstruye un controlador difuso candidato a partir de un cromosoma.

        Args:
            chromosome: Vector de genes reales.

        Returns:
            Clon del controlador base con parámetros de membresía reemplazados.
        """
        params = self.encoder.decode(chromosome)
        controller = self.base_controller.clone()
        controller.set_membership_params(params)
        return controller

    def _optimization_score(self, metrics: PerformanceMetrics) -> float:
        """
        Convierte métricas de simulación en un score escalar para el GA.

        La función combina:

        - ahorro de costo;
        - ahorro de energía;
        - reducción de picos;
        - reducción de variabilidad;
        - penalizaciones por degradación térmica;
        - y una bonificación mínima por mejora de confort.

        El principio dominante es deliberado: una solución no debe ganar solo
        porque ahorra un poco si a cambio empeora de forma apreciable el
        comportamiento térmico.

        Args:
            metrics: Métricas obtenidas al simular el candidato.

        Returns:
            Score escalar mayor es mejor.
        """
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

        # Perder confort respecto al baseline es una señal fuerte de mala calidad.
        comfort_penalty = (comfort_drop / 100.0) * 5.0

        # El ahorro económico es la prioridad principal; energía es secundaria.
        savings_core = 0.65 * cost_saving + 0.25 * energy_saving
        operational_bonus = 0.0
        if cost_saving > 0.0 or energy_saving > 0.0:
            operational_bonus = 0.06 * peak_saving + 0.04 * variability_saving

        # La mejora de confort solo se premia levemente para evitar soluciones
        # que "compren" confort con más gasto.
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
        """
        Registra el mejor candidato observado según criterios de factibilidad.

        El proyecto distingue entre:

        - score numérico;
        - y solución defendible.

        Por eso se marca como factible a un candidato que al menos no empeora
        confort, costo y energía frente al baseline. Esa factibilidad participa
        en el criterio de selección del `best_candidate`.
        """
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
        """Calcula mejora relativa; positiva si el candidato mejora respecto a la referencia."""
        if abs(reference_value) < 1e-9:
            return 0.0
        return (reference_value - candidate_value) / abs(reference_value)

    @property
    def evaluations_count(self) -> int:
        return self._eval_count

    def reset_counter(self):
        self._eval_count = 0

    def get_best_candidate(self) -> Optional[CandidateRecord]:
        """Retorna el mejor candidato factible observado durante toda la búsqueda."""
        return self.best_candidate
