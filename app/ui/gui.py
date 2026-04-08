# ==============================================================================
# gui.py - Interfaz gráfica profesional con PyQt5
# ==============================================================================
"""
Interfaz gráfica completa del sistema de gestión energética inteligente.
Permite configurar, ejecutar simulaciones, optimizar con GA y visualizar
resultados de forma interactiva.

Características:
- Panel de configuración con parámetros editables
- Ejecución de simulación base y optimización GA
- Pestañas de gráficos con matplotlib embebido
- Tabla de métricas y comparación
- Exportación de resultados
- Barra de progreso y log de actividad
"""

import sys
import os
import traceback
from typing import Optional

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QComboBox, QGroupBox, QTextEdit, QProgressBar,
    QSplitter, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QStatusBar, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController
from app.simulation.simulator import Simulator, SimulationResult
from app.simulation.metrics import calculate_metrics, PerformanceMetrics
from app.simulation.scenario_generator import AVAILABLE_SCENARIOS
from app.genetic.optimizer import GeneticOptimizer, OptimizationResult
from app.genetic.evaluation import ComparativeEvaluation
from app.visualization import plots, dashboard
from app.visualization.report_export import (
    export_csv, export_plots_png, generate_html_report
)


# ==============================================================================
# Hilos de trabajo para no bloquear la GUI
# ==============================================================================

class SimulationWorker(QThread):
    """Hilo para ejecutar simulación sin bloquear la interfaz."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)  # (SimulationResult, PerformanceMetrics)
    error = pyqtSignal(str)

    def __init__(self, config: AppConfig, controller: FuzzyController, label: str = "base"):
        super().__init__()
        self.config = config
        self.controller = controller
        self.label = label

    def run(self):
        try:
            sim = Simulator(self.config)
            result = sim.run(
                self.controller.get_controller_function(),
                label=self.label,
                progress_callback=lambda s, t: self.progress.emit(s, t)
            )
            metrics = calculate_metrics(
                result.data, self.config.simulation, self.config.metrics
            )
            self.finished.emit(result, metrics)
        except Exception as e:
            self.error.emit(f"Error en simulación: {str(e)}\n{traceback.format_exc()}")


class OptimizationWorker(QThread):
    """Hilo para ejecutar optimización genética."""
    progress = pyqtSignal(int, int, float)  # gen, total, best_fitness
    finished = pyqtSignal(object, object)  # (OptimizationResult, FuzzyController optimizado)
    error = pyqtSignal(str)

    def __init__(self, config: AppConfig, base_controller: FuzzyController):
        super().__init__()
        self.config = config
        self.base_controller = base_controller
        self.optimizer = None

    def run(self):
        try:
            self.optimizer = GeneticOptimizer(self.config, self.base_controller)
            result = self.optimizer.optimize(
                progress_callback=lambda g, t, f: self.progress.emit(g, t, f)
            )
            opt_controller = self.optimizer.decode_best(result)
            self.finished.emit(result, opt_controller)
        except Exception as e:
            self.error.emit(f"Error en optimización: {str(e)}\n{traceback.format_exc()}")

    def stop(self):
        if self.optimizer:
            self.optimizer.stop()


# ==============================================================================
# Widget de gráfico matplotlib embebido
# ==============================================================================

class MplCanvas(QWidget):
    """Widget que contiene una figura matplotlib embebida."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6), facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background: #16213e; color: #ccc;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def update_figure(self, new_figure: Figure):
        """Reemplaza la figura actual con una nueva."""
        # Limpiar
        layout = self.layout()
        layout.removeWidget(self.toolbar)
        layout.removeWidget(self.canvas)
        self.toolbar.deleteLater()
        self.canvas.deleteLater()

        self.figure = new_figure
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background: #16213e; color: #ccc;")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.canvas.draw()


# ==============================================================================
# Ventana principal
# ==============================================================================

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏠 Sistema Inteligente de Gestión Energética Residencial")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Estado
        self.config = AppConfig()
        self.base_controller: Optional[FuzzyController] = None
        self.optimized_controller: Optional[FuzzyController] = None
        self.base_result: Optional[SimulationResult] = None
        self.optimized_result: Optional[SimulationResult] = None
        self.base_metrics: Optional[PerformanceMetrics] = None
        self.optimized_metrics: Optional[PerformanceMetrics] = None
        self.ga_result: Optional[OptimizationResult] = None

        # Workers
        self._sim_worker: Optional[SimulationWorker] = None
        self._opt_worker: Optional[OptimizationWorker] = None

        self._apply_dark_theme()
        self._build_ui()
        self._init_controller()
        self._log("Sistema iniciado. Configure los parámetros y ejecute la simulación.")

    # ------------------------------------------------------------------
    # Tema oscuro
    # ------------------------------------------------------------------
    def _apply_dark_theme(self):
        """Aplica tema oscuro profesional a toda la aplicación."""
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a1a; }
            QWidget { background-color: #0a0a1a; color: #e0e0e0; font-family: 'Segoe UI'; }
            QGroupBox {
                background-color: #16213e;
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 20px;
                font-weight: bold;
                color: #53d8fb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #e94560;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #e94560;
                color: white;
            }
            QPushButton:disabled {
                background-color: #1a1a2e;
                color: #555;
                border-color: #333;
            }
            QPushButton#btn_run_base {
                background-color: #1b5e20;
                border-color: #66bb6a;
            }
            QPushButton#btn_run_base:hover { background-color: #2e7d32; }
            QPushButton#btn_run_ga {
                background-color: #4a148c;
                border-color: #ab47bc;
            }
            QPushButton#btn_run_ga:hover { background-color: #6a1b9a; }
            QPushButton#btn_compare {
                background-color: #e65100;
                border-color: #ff7043;
            }
            QPushButton#btn_compare:hover { background-color: #f4511e; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #16213e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 26px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #53d8fb;
            }
            QTabWidget::pane {
                background-color: #0a0a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #16213e;
                color: #aaa;
                padding: 8px 18px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #0f3460;
                color: #53d8fb;
                border-bottom: 2px solid #e94560;
            }
            QTextEdit {
                background-color: #0d1117;
                color: #8b949e;
                border: 1px solid #333;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QProgressBar {
                background-color: #16213e;
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                color: #eee;
                font-size: 11px;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #e94560;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #16213e;
                color: #e0e0e0;
                gridline-color: #333;
                border: 1px solid #333;
                border-radius: 4px;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #0f3460;
                color: #53d8fb;
                padding: 6px;
                border: 1px solid #333;
                font-weight: bold;
                font-size: 11px;
            }
            QLabel { font-size: 12px; }
            QStatusBar { background: #16213e; color: #888; }
            QSplitter::handle { background-color: #333; width: 2px; }
        """)

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Construye toda la interfaz gráfica."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Splitter principal: panel izquierdo (config) | panel derecho (resultados)
        splitter = QSplitter(Qt.Horizontal)

        # --- Panel izquierdo: Configuración y acciones ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Título
        title = QLabel("⚡ Gestión Energética IA")
        title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        title.setStyleSheet("color: #e94560; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Grupo: Simulación
        sim_group = self._build_sim_params_group()
        left_layout.addWidget(sim_group)

        # Grupo: Algoritmo Genético
        ga_group = self._build_ga_params_group()
        left_layout.addWidget(ga_group)

        # Botones de acción
        actions_group = self._build_actions_group()
        left_layout.addWidget(actions_group)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("📋 Log de Actividad")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_panel.setMaximumWidth(360)
        left_panel.setMinimumWidth(300)

        # --- Panel derecho: Resultados ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self._build_tabs()
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([340, 1100])

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Listo")

    def _build_sim_params_group(self) -> QGroupBox:
        """Grupo de parámetros de simulación."""
        group = QGroupBox("🔧 Parámetros de Simulación")
        layout = QGridLayout(group)

        # Duración
        layout.addWidget(QLabel("Duración (horas):"), 0, 0)
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(12, 336)
        self.spin_duration.setValue(72)
        self.spin_duration.setSingleStep(24)
        layout.addWidget(self.spin_duration, 0, 1)

        # Semilla
        layout.addWidget(QLabel("Semilla:"), 1, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        layout.addWidget(self.spin_seed, 1, 1)

        # Temperatura objetivo
        layout.addWidget(QLabel("T. Objetivo (°C):"), 2, 0)
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(16.0, 28.0)
        self.spin_target.setValue(22.0)
        self.spin_target.setSingleStep(0.5)
        layout.addWidget(self.spin_target, 2, 1)

        # Rango de confort
        layout.addWidget(QLabel("Rango confort (±°C):"), 3, 0)
        self.spin_comfort = QDoubleSpinBox()
        self.spin_comfort.setRange(0.5, 5.0)
        self.spin_comfort.setValue(2.0)
        self.spin_comfort.setSingleStep(0.5)
        layout.addWidget(self.spin_comfort, 3, 1)

        # Escenario
        layout.addWidget(QLabel("Escenario:"), 4, 0)
        self.combo_scenario = QComboBox()
        self.combo_scenario.addItems(AVAILABLE_SCENARIOS)
        self.combo_scenario.setCurrentText('verano')
        layout.addWidget(self.combo_scenario, 4, 1)

        return group

    def _build_ga_params_group(self) -> QGroupBox:
        """Grupo de parámetros del algoritmo genético."""
        group = QGroupBox("🧬 Algoritmo Genético")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Población:"), 0, 0)
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 200)
        self.spin_pop.setValue(60)
        self.spin_pop.setSingleStep(10)
        layout.addWidget(self.spin_pop, 0, 1)

        layout.addWidget(QLabel("Generaciones:"), 1, 0)
        self.spin_gens = QSpinBox()
        self.spin_gens.setRange(5, 200)
        self.spin_gens.setValue(40)
        self.spin_gens.setSingleStep(5)
        layout.addWidget(self.spin_gens, 1, 1)

        layout.addWidget(QLabel("P. Cruce:"), 2, 0)
        self.spin_crossover = QDoubleSpinBox()
        self.spin_crossover.setRange(0.1, 1.0)
        self.spin_crossover.setValue(0.85)
        self.spin_crossover.setSingleStep(0.05)
        layout.addWidget(self.spin_crossover, 2, 1)

        layout.addWidget(QLabel("P. Mutación:"), 3, 0)
        self.spin_mutation = QDoubleSpinBox()
        self.spin_mutation.setRange(0.01, 0.5)
        self.spin_mutation.setValue(0.12)
        self.spin_mutation.setSingleStep(0.01)
        layout.addWidget(self.spin_mutation, 3, 1)

        return group

    def _build_actions_group(self) -> QGroupBox:
        """Grupo de botones de acción."""
        group = QGroupBox("▶️ Acciones")
        layout = QVBoxLayout(group)

        self.btn_run_base = QPushButton("▶  Ejecutar Simulación Base")
        self.btn_run_base.setObjectName("btn_run_base")
        self.btn_run_base.clicked.connect(self._on_run_base)
        layout.addWidget(self.btn_run_base)

        self.btn_run_ga = QPushButton("🧬 Ejecutar Optimización GA")
        self.btn_run_ga.setObjectName("btn_run_ga")
        self.btn_run_ga.clicked.connect(self._on_run_ga)
        self.btn_run_ga.setEnabled(False)
        layout.addWidget(self.btn_run_ga)

        self.btn_compare = QPushButton("🔄 Comparar Resultados")
        self.btn_compare.setObjectName("btn_compare")
        self.btn_compare.clicked.connect(self._on_compare)
        self.btn_compare.setEnabled(False)
        layout.addWidget(self.btn_compare)

        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)

        self.btn_export = QPushButton("💾 Exportar Reporte HTML")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        self.btn_export_csv = QPushButton("📄 Exportar CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv.setEnabled(False)
        layout.addWidget(self.btn_export_csv)

        return group

    def _build_tabs(self):
        """Construye las pestañas de resultados."""
        # Tab 1: Dashboard base
        self.canvas_dashboard = MplCanvas()
        self.tabs.addTab(self.canvas_dashboard, "📊 Dashboard")

        # Tab 2: Temperaturas
        self.canvas_temp = MplCanvas()
        self.tabs.addTab(self.canvas_temp, "🌡️ Temperaturas")

        # Tab 3: HVAC
        self.canvas_hvac = MplCanvas()
        self.tabs.addTab(self.canvas_hvac, "❄️ HVAC")

        # Tab 4: Consumo
        self.canvas_consumption = MplCanvas()
        self.tabs.addTab(self.canvas_consumption, "⚡ Consumo")

        # Tab 5: Costo
        self.canvas_cost = MplCanvas()
        self.tabs.addTab(self.canvas_cost, "💰 Costo")

        # Tab 6: Confort
        self.canvas_comfort = MplCanvas()
        self.tabs.addTab(self.canvas_comfort, "😊 Confort")

        # Tab 7: GA Evolution
        self.canvas_ga = MplCanvas()
        self.tabs.addTab(self.canvas_ga, "🧬 GA")

        # Tab 8: Comparación
        self.canvas_compare = MplCanvas()
        self.tabs.addTab(self.canvas_compare, "🔄 Comparación")

        # Tab 9: Métricas tabla
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels([
            'Métrica', 'Base', 'Optimizado', 'Cambio (%)'
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents)
        self.tabs.addTab(self.metrics_table, "📋 Métricas")

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def _init_controller(self):
        """Inicializa el controlador difuso base."""
        self.base_controller = FuzzyController(self.config.fuzzy)
        self._log(f"Controlador difuso inicializado: {self.base_controller}")

    def _update_config(self):
        """Actualiza la configuración desde los widgets de la GUI."""
        self.config.simulation.horizon_hours = self.spin_duration.value()
        self.config.simulation.random_seed = self.spin_seed.value()
        self.config.simulation.target_temperature = self.spin_target.value()
        self.config.simulation.comfort_range = self.spin_comfort.value()
        self.config.simulation.scenario_type = self.combo_scenario.currentText()
        
        self.config.genetic.population_size = self.spin_pop.value()
        self.config.genetic.num_generations = self.spin_gens.value()
        self.config.genetic.crossover_probability = self.spin_crossover.value()
        self.config.genetic.mutation_probability = self.spin_mutation.value()
        self.config.genetic.random_seed = self.spin_seed.value()
        
        # Sincronizar confort
        self.config.metrics.comfort_min = (
            self.config.simulation.target_temperature - self.config.simulation.comfort_range
        )
        self.config.metrics.comfort_max = (
            self.config.simulation.target_temperature + self.config.simulation.comfort_range
        )

    # ------------------------------------------------------------------
    # Acciones
    # ------------------------------------------------------------------
    def _on_run_base(self):
        """Ejecuta simulación base."""
        self._update_config()
        self._init_controller()
        
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._log("Iniciando simulación base...")
        self.statusBar().showMessage("Simulación base en progreso...")

        self._sim_worker = SimulationWorker(self.config, self.base_controller, "base")
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.finished.connect(self._on_base_finished)
        self._sim_worker.error.connect(self._on_error)
        self._sim_worker.start()

    def _on_base_finished(self, result: SimulationResult, metrics: PerformanceMetrics):
        """Callback cuando la simulación base termina."""
        self.base_result = result
        self.base_metrics = metrics

        self._log(f"✅ Simulación base completada: {result.num_steps} pasos")
        self._log(f"   Energía: {metrics.total_energy_kwh:.1f} kWh | "
                  f"Costo: ${metrics.total_cost:.2f} | "
                  f"Confort: {metrics.comfort_percentage:.1f}%")

        # Actualizar gráficos
        self._update_base_plots()
        self._update_metrics_table()

        self.progress_bar.setValue(100)
        self.btn_run_ga.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Simulación base completada ✅")
        self.tabs.setCurrentIndex(0)  # Ir a dashboard

    def _on_run_ga(self):
        """Ejecuta optimización genética."""
        self._update_config()
        
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._log(f"Iniciando optimización GA: {self.config.genetic.population_size} individuos × "
                  f"{self.config.genetic.num_generations} generaciones...")
        self.statusBar().showMessage("Optimización GA en progreso...")

        self._opt_worker = OptimizationWorker(self.config, self.base_controller)
        self._opt_worker.progress.connect(self._on_ga_progress)
        self._opt_worker.finished.connect(self._on_ga_finished)
        self._opt_worker.error.connect(self._on_error)
        self._opt_worker.start()

    def _on_ga_finished(self, ga_result: OptimizationResult, opt_controller: FuzzyController):
        """Callback cuando la optimización GA termina."""
        self.ga_result = ga_result
        self.optimized_controller = opt_controller

        self._log(f"✅ Optimización completada en {ga_result.total_time_seconds:.1f}s")
        self._log(f"   Mejor fitness: {ga_result.best_fitness:.4f} | "
                  f"Evaluaciones: {ga_result.total_evaluations}")

        # Ejecutar simulación con controlador optimizado
        self._log("Ejecutando simulación con controlador optimizado...")
        sim = Simulator(self.config)
        self.optimized_result = sim.run(
            opt_controller.get_controller_function(), label="optimizado"
        )
        self.optimized_metrics = calculate_metrics(
            self.optimized_result.data, self.config.simulation, self.config.metrics
        )

        self._log(f"   Energía opt: {self.optimized_metrics.total_energy_kwh:.1f} kWh | "
                  f"Costo: ${self.optimized_metrics.total_cost:.2f} | "
                  f"Confort: {self.optimized_metrics.comfort_percentage:.1f}%")

        # Actualizar gráfico GA
        self._update_ga_plot()
        self._update_metrics_table()

        self.progress_bar.setValue(100)
        self.btn_compare.setEnabled(True)
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Optimización GA completada ✅")
        self.tabs.setCurrentIndex(6)  # Ir a tab GA

    def _on_compare(self):
        """Genera comparación base vs optimizado."""
        if not self.base_result or not self.optimized_result:
            self._log("⚠️ Necesita simulación base y optimizada para comparar.")
            return

        self._log("Generando comparación...")
        self._update_comparison_plots()
        self._update_metrics_table()
        self.tabs.setCurrentIndex(7)  # Ir a tab comparación
        self._log("✅ Comparación generada.")
        self.statusBar().showMessage("Comparación lista ✅")

    def _on_export(self):
        """Exporta reporte HTML."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Reporte HTML", 
            os.path.join(self.config.paths.outputs_dir, "reporte.html"),
            "HTML (*.html)"
        )
        if filepath:
            try:
                ga_hist = self.ga_result.get_fitness_history() if self.ga_result else None
                ga_avg = self.ga_result.get_avg_fitness_history() if self.ga_result else None
                
                generate_html_report(
                    self.config,
                    self.base_result, self.optimized_result,
                    self.base_metrics, self.optimized_metrics,
                    ga_hist, ga_avg,
                    output_path=filepath
                )
                self._log(f"✅ Reporte exportado: {filepath}")
                self.statusBar().showMessage(f"Reporte guardado en {filepath}")
            except Exception as e:
                self._on_error(f"Error exportando: {str(e)}")

    def _on_export_csv(self):
        """Exporta datos a CSV."""
        dirpath = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de exportación",
            self.config.paths.outputs_dir
        )
        if dirpath:
            try:
                if self.base_result:
                    path = export_csv(self.base_result, dirpath)
                    self._log(f"📄 CSV base: {path}")
                if self.optimized_result:
                    path = export_csv(self.optimized_result, dirpath)
                    self._log(f"📄 CSV optimizado: {path}")
                self.statusBar().showMessage("CSV exportados ✅")
            except Exception as e:
                self._on_error(f"Error exportando CSV: {str(e)}")

    # ------------------------------------------------------------------
    # Progreso y errores
    # ------------------------------------------------------------------
    def _on_sim_progress(self, step: int, total: int):
        pct = int((step / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)

    def _on_ga_progress(self, gen: int, total: int, best_fitness: float):
        pct = int((gen / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.statusBar().showMessage(
            f"GA: Generación {gen}/{total} | Mejor fitness: {best_fitness:.4f}"
        )

    def _on_error(self, msg: str):
        self._log(f"❌ {msg}")
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Error", msg)

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_run_base.setEnabled(enabled)
        if enabled and self.base_result:
            self.btn_run_ga.setEnabled(True)
        elif not enabled:
            self.btn_run_ga.setEnabled(False)
        if enabled and self.optimized_result:
            self.btn_compare.setEnabled(True)
        elif not enabled:
            self.btn_compare.setEnabled(False)

    def _log(self, msg: str):
        self.log_text.append(msg)
        # Auto-scroll
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    # ------------------------------------------------------------------
    # Actualización de gráficos
    # ------------------------------------------------------------------
    def _update_base_plots(self):
        """Actualiza todos los gráficos con datos de simulación base."""
        if not self.base_result:
            return

        df = self.base_result.data
        target = self.config.simulation.target_temperature
        comfort = self.config.simulation.comfort_range

        try:
            # Dashboard
            fig = dashboard.create_simulation_dashboard(
                df, target, comfort, "Dashboard — Simulación Base"
            )
            self.canvas_dashboard.update_figure(fig)

            # Temperaturas
            fig = plots.plot_temperatures(df, target, comfort)
            self.canvas_temp.update_figure(fig)

            # HVAC
            fig = plots.plot_hvac_level(df)
            self.canvas_hvac.update_figure(fig)

            # Consumo
            fig = plots.plot_consumption(df)
            self.canvas_consumption.update_figure(fig)

            # Costo
            fig = plots.plot_cumulative_cost(df)
            self.canvas_cost.update_figure(fig)

            # Confort
            fig = plots.plot_comfort(df, target)
            self.canvas_comfort.update_figure(fig)

        except Exception as e:
            self._log(f"⚠️ Error actualizando gráficos: {e}")

    def _update_ga_plot(self):
        """Actualiza el gráfico de evolución del GA."""
        if not self.ga_result:
            return
        try:
            fig = plots.plot_ga_evolution(
                self.ga_result.get_fitness_history(),
                self.ga_result.get_avg_fitness_history()
            )
            self.canvas_ga.update_figure(fig)
        except Exception as e:
            self._log(f"⚠️ Error en gráfico GA: {e}")

    def _update_comparison_plots(self):
        """Actualiza gráficos de comparación."""
        if not self.base_result or not self.optimized_result:
            return
        try:
            target = self.config.simulation.target_temperature
            comfort = self.config.simulation.comfort_range

            fig = plots.plot_comparison(
                self.base_result.data, self.optimized_result.data,
                target, comfort
            )
            self.canvas_compare.update_figure(fig)
        except Exception as e:
            self._log(f"⚠️ Error en comparación: {e}")

    def _update_metrics_table(self):
        """Actualiza la tabla de métricas."""
        if not self.base_metrics:
            return

        base_dict = self.base_metrics.to_dict()
        opt_dict = self.optimized_metrics.to_dict() if self.optimized_metrics else {}

        self.metrics_table.setRowCount(len(base_dict))

        for row, (key, base_val) in enumerate(base_dict.items()):
            # Métrica
            self.metrics_table.setItem(row, 0, QTableWidgetItem(key))
            # Base
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{base_val}"))
            # Optimizado
            opt_val = opt_dict.get(key, '-')
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{opt_val}"))
            # Cambio
            if isinstance(opt_val, (int, float)) and isinstance(base_val, (int, float)) and base_val != 0:
                change = ((opt_val - base_val) / abs(base_val)) * 100
                item = QTableWidgetItem(f"{change:+.1f}%")
                if change < 0:
                    item.setForeground(QColor('#66bb6a'))
                else:
                    item.setForeground(QColor('#ef5350'))
                self.metrics_table.setItem(row, 3, item)
            else:
                self.metrics_table.setItem(row, 3, QTableWidgetItem('-'))


# ==============================================================================
# Función de lanzamiento
# ==============================================================================

def launch_gui():
    """Lanza la interfaz gráfica de la aplicación."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Paleta oscura
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('#0a0a1a'))
    palette.setColor(QPalette.WindowText, QColor('#e0e0e0'))
    palette.setColor(QPalette.Base, QColor('#16213e'))
    palette.setColor(QPalette.AlternateBase, QColor('#1a1a2e'))
    palette.setColor(QPalette.Text, QColor('#e0e0e0'))
    palette.setColor(QPalette.Button, QColor('#0f3460'))
    palette.setColor(QPalette.ButtonText, QColor('#e0e0e0'))
    palette.setColor(QPalette.Highlight, QColor('#e94560'))
    palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()
