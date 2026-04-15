from typing import Dict, Tuple, Optional
import pandas as pd

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController
from app.simulation.simulator import Simulator, SimulationResult
from app.simulation.metrics import (
    calculate_metrics, compare_metrics, PerformanceMetrics
)


class ComparativeEvaluation:
    """
    Ejecuta y conserva comparaciones operativas entre dos controladores.

    Este objeto encapsula el flujo de evaluación posterior a la optimización:

    - correr baseline;
    - correr versión optimizada;
    - calcular métricas de ambos;
    - construir estructuras comparables para GUI y reportes.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.base_result: Optional[SimulationResult] = None
        self.optimized_result: Optional[SimulationResult] = None
        self.base_metrics: Optional[PerformanceMetrics] = None
        self.optimized_metrics: Optional[PerformanceMetrics] = None
        self.comparison: Optional[Dict] = None

    def run_base(self, controller: FuzzyController,
                 progress_callback=None) -> SimulationResult:
        """
        Ejecuta la simulación del controlador base y guarda sus métricas.

        Returns:
            Resultado completo de simulación para el baseline.
        """
        simulator = Simulator(self.config)
        self.base_result = simulator.run(
            controller.get_controller_function(),
            label="base",
            progress_callback=progress_callback
        )
        self.base_metrics = calculate_metrics(
            self.base_result.data,
            self.config.simulation,
            self.config.metrics
        )
        return self.base_result

    def run_optimized(self, controller: FuzzyController,
                      progress_callback=None) -> SimulationResult:
        """
        Ejecuta la simulación del controlador optimizado y guarda sus métricas.

        Returns:
            Resultado completo de simulación para el controlador afinado.
        """
        simulator = Simulator(self.config)
        self.optimized_result = simulator.run(
            controller.get_controller_function(),
            label="optimizado",
            progress_callback=progress_callback
        )
        self.optimized_metrics = calculate_metrics(
            self.optimized_result.data,
            self.config.simulation,
            self.config.metrics
        )
        return self.optimized_result

    def compare(self) -> Dict:
        """
        Construye la comparación métrica final entre ambos controladores.

        Returns:
            Diccionario con métricas base, optimizadas y cambio porcentual.
        """
        if self.base_metrics is None or self.optimized_metrics is None:
            raise RuntimeError("Debe ejecutar ambas simulaciones antes de comparar.")
        
        self.comparison = compare_metrics(self.base_metrics, self.optimized_metrics)
        return self.comparison

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Convierte la comparación en una tabla adecuada para visualización.

        Returns:
            `DataFrame` con una fila por métrica y columnas para base,
            optimizado y variación porcentual.
        """
        if self.comparison is None:
            self.compare()
        
        rows = []
        for metric_name, values in self.comparison.items():
            rows.append({
                'Métrica': metric_name,
                'Base': values['base'],
                'Optimizado': values['optimizado'],
                'Cambio (%)': values['cambio_%'],
            })
        
        return pd.DataFrame(rows)

    def get_summary_text(self) -> str:
        """
        Genera un resumen textual legible para consola o reportes simples.

        Returns:
            Texto multilínea con el resumen comparativo.
        """
        if self.comparison is None:
            self.compare()
        
        lines = [
            "=" * 60,
            "  COMPARACIÓN: CONTROLADOR BASE vs OPTIMIZADO",
            "=" * 60,
        ]
        
        for metric_name, values in self.comparison.items():
            change = values['cambio_%']
            arrow = "+" if change > 0 else "-" if change < 0 else "="
            lines.append(
                f"  {metric_name:<35} "
                f"Base: {values['base']:>10.3f}  "
                f"Opt: {values['optimizado']:>10.3f}  "
                f"{arrow} {abs(change):.1f}%"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
