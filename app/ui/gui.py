# ==============================================================================
# gui.py - Interfaz grafica profesional con PyQt5
# ==============================================================================
"""
Interfaz grafica completa del sistema de gestion energetica inteligente.

Caracteristicas principales:
- Panel de configuracion con parametros editables
- Selector de dispositivo (HVAC / Refrigerador)
- Pestana de Inferencia Difusa interactiva con entradas linguisticas
- Visualizacion de funciones de pertenencia
- Visualizacion de reglas activadas
- Ejecucion de simulacion base y optimizacion GA
- Comparacion base vs optimizado en TODAS las vistas
- Exportacion de resultados

Sin emojis. Se usan iconos textuales donde corresponde.
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
    QMessageBox, QFrame, QSizePolicy, QStatusBar, QHeaderView,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController, InferenceDetail
from app.fuzzy.linguistic import LinguisticInput, LinguisticOutput
from app.simulation.simulator import Simulator, SimulationResult
from app.simulation.metrics import calculate_metrics, PerformanceMetrics
from app.simulation.scenario_generator import AVAILABLE_SCENARIOS
from app.simulation.devices import AVAILABLE_DEVICES, get_hvac_config, get_refrigerator_config
from app.genetic.optimizer import GeneticOptimizer, OptimizationResult
from app.visualization import plots
from app.visualization import fuzzy_plots
from app.visualization import dashboard
from app.visualization.report_export import (
    export_csv, generate_html_report
)


# ==============================================================================
# Hilos de trabajo
# ==============================================================================

class SimulationWorker(QThread):
    """Hilo para ejecutar simulacion."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, config, controller, label="base"):
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
            self.error.emit(f"Error en simulacion: {str(e)}\n{traceback.format_exc()}")


class OptimizationWorker(QThread):
    """Hilo para optimizacion genetica."""
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, config, base_controller):
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
            self.error.emit(f"Error en optimizacion: {str(e)}\n{traceback.format_exc()}")

    def stop(self):
        if self.optimizer:
            self.optimizer.stop()


# ==============================================================================
# Widget de grafico matplotlib
# ==============================================================================

class MplCanvas(QWidget):
    """Widget con figura matplotlib embebida."""

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
        """Reemplaza la figura actual."""
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
    """Ventana principal de la aplicacion."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Inteligente de Gestion Energetica Residencial")
        self.setMinimumSize(1320, 820)
        self.resize(1480, 920)

        # Estado
        self.config = AppConfig()
        self.base_controller: Optional[FuzzyController] = None
        self.optimized_controller: Optional[FuzzyController] = None
        self.base_result: Optional[SimulationResult] = None
        self.optimized_result: Optional[SimulationResult] = None
        self.base_metrics: Optional[PerformanceMetrics] = None
        self.optimized_metrics: Optional[PerformanceMetrics] = None
        self.ga_result: Optional[OptimizationResult] = None

        # Componentes linguisticos
        self.linguistic_input = LinguisticInput()
        self.linguistic_output = LinguisticOutput()

        # Workers
        self._sim_worker: Optional[SimulationWorker] = None
        self._opt_worker: Optional[OptimizationWorker] = None

        self._apply_dark_theme()
        self._build_ui()
        self._init_controller()
        self._log("Sistema iniciado. Configure los parametros y ejecute la simulacion.")

    # ------------------------------------------------------------------
    # Tema oscuro
    # ------------------------------------------------------------------
    def _apply_dark_theme(self):
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
            QPushButton:hover { background-color: #e94560; color: white; }
            QPushButton:disabled { background-color: #1a1a2e; color: #555; border-color: #333; }
            QPushButton#btn_run_base { background-color: #1b5e20; border-color: #66bb6a; }
            QPushButton#btn_run_base:hover { background-color: #2e7d32; }
            QPushButton#btn_run_ga { background-color: #4a148c; border-color: #ab47bc; }
            QPushButton#btn_run_ga:hover { background-color: #6a1b9a; }
            QPushButton#btn_infer { background-color: #01579b; border-color: #4fc3f7; }
            QPushButton#btn_infer:hover { background-color: #0277bd; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #16213e; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px 8px; min-height: 26px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #53d8fb; }
            QTabWidget::pane { background-color: #0a0a1a; border: 1px solid #333; border-radius: 4px; }
            QTabBar::tab {
                background: #16213e; color: #aaa; padding: 8px 18px;
                margin-right: 2px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; font-size: 12px;
            }
            QTabBar::tab:selected { background: #0f3460; color: #53d8fb; border-bottom: 2px solid #e94560; }
            QTextEdit {
                background-color: #0d1117; color: #8b949e; border: 1px solid #333;
                border-radius: 4px; font-family: 'Consolas', monospace; font-size: 11px;
            }
            QProgressBar {
                background-color: #16213e; border: 1px solid #333; border-radius: 4px;
                text-align: center; color: #eee; font-size: 11px; min-height: 20px;
            }
            QProgressBar::chunk { background-color: #e94560; border-radius: 3px; }
            QTableWidget { background-color: #16213e; color: #e0e0e0; gridline-color: #333; border: 1px solid #333; border-radius: 4px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { background-color: #0f3460; color: #53d8fb; padding: 6px; border: 1px solid #333; font-weight: bold; font-size: 11px; }
            QLabel { font-size: 12px; }
            QStatusBar { background: #16213e; color: #888; }
            QSplitter::handle { background-color: #333; width: 2px; }
            QScrollArea { background-color: #0a0a1a; border: none; }
        """)

    # ------------------------------------------------------------------
    # Construccion de la interfaz
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # --- Panel izquierdo: Configuracion y acciones ---
        left_panel = self._build_left_panel()
        left_panel.setMaximumWidth(370)
        left_panel.setMinimumWidth(310)

        # --- Panel derecho: Pestanas de resultados ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        self._build_tabs()
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1130])
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Listo")

    def _build_left_panel(self) -> QWidget:
        """Construye el panel lateral izquierdo completo."""
        panel = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)

        # Titulo
        title = QLabel("Gestion Energetica IA")
        title.setFont(QFont('Segoe UI', 15, QFont.Bold))
        title.setStyleSheet("color: #e94560; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Selector de dispositivo
        layout.addWidget(self._build_device_selector())

        # Parametros de simulacion
        layout.addWidget(self._build_sim_params_group())

        # Parametros GA
        layout.addWidget(self._build_ga_params_group())

        # Botones de accion
        layout.addWidget(self._build_actions_group())

        # Progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("Log de Actividad")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(130)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()
        scroll.setWidget(inner)

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
        return panel

    def _build_device_selector(self) -> QGroupBox:
        """Selector de dispositivo controlado."""
        group = QGroupBox("Dispositivo Controlado")
        layout = QGridLayout(group)
        layout.addWidget(QLabel("Dispositivo:"), 0, 0)
        self.combo_device = QComboBox()
        self.combo_device.addItems([
            "HVAC (Climatizacion)",
            "Refrigerador"
        ])
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.combo_device, 0, 1)
        return group

    def _build_sim_params_group(self) -> QGroupBox:
        group = QGroupBox("Parametros de Simulacion")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Duracion (horas):"), 0, 0)
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(12, 336)
        self.spin_duration.setValue(72)
        self.spin_duration.setSingleStep(24)
        layout.addWidget(self.spin_duration, 0, 1)

        layout.addWidget(QLabel("Semilla:"), 1, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        layout.addWidget(self.spin_seed, 1, 1)

        layout.addWidget(QLabel("T. Objetivo (C):"), 2, 0)
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(-10.0, 30.0)
        self.spin_target.setValue(22.0)
        self.spin_target.setSingleStep(0.5)
        layout.addWidget(self.spin_target, 2, 1)

        layout.addWidget(QLabel("Rango confort (+/-C):"), 3, 0)
        self.spin_comfort = QDoubleSpinBox()
        self.spin_comfort.setRange(0.5, 5.0)
        self.spin_comfort.setValue(2.0)
        self.spin_comfort.setSingleStep(0.5)
        layout.addWidget(self.spin_comfort, 3, 1)

        layout.addWidget(QLabel("Escenario:"), 4, 0)
        self.combo_scenario = QComboBox()
        self.combo_scenario.addItems(AVAILABLE_SCENARIOS)
        self.combo_scenario.setCurrentText('verano')
        layout.addWidget(self.combo_scenario, 4, 1)

        return group

    def _build_ga_params_group(self) -> QGroupBox:
        group = QGroupBox("Algoritmo Genetico (Optimizador)")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Poblacion:"), 0, 0)
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

        layout.addWidget(QLabel("P. Mutacion:"), 3, 0)
        self.spin_mutation = QDoubleSpinBox()
        self.spin_mutation.setRange(0.01, 0.5)
        self.spin_mutation.setValue(0.12)
        self.spin_mutation.setSingleStep(0.01)
        layout.addWidget(self.spin_mutation, 3, 1)

        return group

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("Acciones")
        layout = QVBoxLayout(group)

        self.btn_run_base = QPushButton("[>] Ejecutar Control Difuso (Por Defecto)")
        self.btn_run_base.setObjectName("btn_run_base")
        self.btn_run_base.clicked.connect(self._on_run_base)
        layout.addWidget(self.btn_run_base)

        self.btn_run_ga = QPushButton("[GA] Afinar Parámetros Difusos (Genético)")
        self.btn_run_ga.setObjectName("btn_run_ga")
        self.btn_run_ga.clicked.connect(self._on_run_ga)
        self.btn_run_ga.setEnabled(False)
        layout.addWidget(self.btn_run_ga)

        self.btn_compare = QPushButton("[<>] Comparar Control Difuso: Defecto vs Afinado")
        self.btn_compare.clicked.connect(self._on_compare)
        self.btn_compare.setEnabled(False)
        layout.addWidget(self.btn_compare)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)

        self.btn_export = QPushButton("[H] Exportar Reporte HTML")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        self.btn_export_csv = QPushButton("[C] Exportar CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv.setEnabled(False)
        layout.addWidget(self.btn_export_csv)

        return group

    # ------------------------------------------------------------------
    # Pestanas
    # ------------------------------------------------------------------
    def _build_tabs(self):
        """Construye todas las pestanas de resultados."""

        # Tab 0: Dashboard (Base only)
        self.canvas_dashboard = MplCanvas()
        self.tabs.addTab(self.canvas_dashboard, "Dashboard")

        # Tab 1: Temperaturas
        self.canvas_temp = MplCanvas()
        self.tabs.addTab(self.canvas_temp, "Temperaturas")

        # Tab 2: Humedad
        self.canvas_humidity = MplCanvas()
        self.tabs.addTab(self.canvas_humidity, "Humedad")

        # Tab 3: HVAC
        self.canvas_hvac = MplCanvas()
        self.tabs.addTab(self.canvas_hvac, "Control")

        # Tab 4: Consumo
        self.canvas_consumption = MplCanvas()
        self.tabs.addTab(self.canvas_consumption, "Consumo")

        # Tab 5: Costo
        self.canvas_cost = MplCanvas()
        self.tabs.addTab(self.canvas_cost, "Costo")

        # Tab 6: Confort
        self.canvas_comfort = MplCanvas()
        self.tabs.addTab(self.canvas_comfort, "Confort")

        # Tab 7: Analisis Difuso (Por Hora)
        self.tabs.addTab(self._build_hourly_inference_tab(), "Analisis Difuso")

        # Tab 8: Optimizacion GA
        self.canvas_ga = MplCanvas()
        self.tabs.addTab(self.canvas_ga, "Optimizacion GA")

        # Tab 9: Funciones de Pertenencia
        self.canvas_mf = MplCanvas()
        self.tabs.addTab(self.canvas_mf, "Funciones de Pertenencia")

        # Tab 10: Comparacion global
        self.canvas_compare = MplCanvas()
        self.tabs.addTab(self.canvas_compare, "Comparacion Global")

        # Tab 11: Metricas tabla
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels([
            'Metrica', 'Base', 'Optimizado', 'Cambio (%)'
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in [1, 2, 3]:
            self.metrics_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.tabs.addTab(self.metrics_table, "Metricas")

    def _build_hourly_inference_tab(self) -> QWidget:
        """
        Pestaña de Análisis Difuso interactivo basado en la simulación real.
        Permite usar un slider para ver hora por hora las reglas activadas.
        """
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Controles superiores
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_panel.setMaximumHeight(60)

        self.lbl_hour = QLabel("Seleccione la Hora de Simulación (0 h):")
        self.lbl_hour.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.lbl_hour.setStyleSheet("color: #53d8fb;")
        control_layout.addWidget(self.lbl_hour)

        from PyQt5.QtWidgets import QSlider
        self.slider_hour = QSlider(Qt.Horizontal)
        self.slider_hour.setMinimum(0)
        self.slider_hour.setMaximum(72)
        self.slider_hour.setValue(0)
        self.slider_hour.setTickPosition(QSlider.TicksBelow)
        self.slider_hour.setTickInterval(6)
        self.slider_hour.setEnabled(False) # Activar tras simular base
        self.slider_hour.valueChanged.connect(self._on_hour_changed)
        control_layout.addWidget(self.slider_hour, stretch=1)
        
        main_layout.addWidget(control_panel)

        # Tablas y resultados
        middle_panel = QWidget()
        middle_layout = QHBoxLayout(middle_panel)
        
        # Tabla de reglas
        rules_group = QGroupBox("Reglas Activadas en esta Hora")
        rules_layout = QVBoxLayout(rules_group)
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(2)
        self.rules_table.setHorizontalHeaderLabels(['Regla Difusa', 'Fuerza de Activacion'])
        self.rules_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rules_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        rules_layout.addWidget(self.rules_table)
        middle_layout.addWidget(rules_group, stretch=1)
        
        # Inputs y Output (Display numerico)
        info_group = QGroupBox("Estado del Entorno")
        info_layout = QVBoxLayout(info_group)
        self.lbl_fuzzy_inputs = QLabel("Sin datos. Ejecute la simulación base.")
        self.lbl_fuzzy_inputs.setWordWrap(True)
        self.lbl_fuzzy_inputs.setStyleSheet("font-size: 13px; color: #aaa;")
        info_layout.addWidget(self.lbl_fuzzy_inputs)
        
        self.lbl_fuzzy_output = QLabel("--")
        self.lbl_fuzzy_output.setFont(QFont('Segoe UI', 14, QFont.Bold))
        self.lbl_fuzzy_output.setStyleSheet("color: #e94560; margin-top: 10px;")
        info_layout.addWidget(self.lbl_fuzzy_output)
        info_layout.addStretch()
        middle_layout.addWidget(info_group)

        main_layout.addWidget(middle_panel, stretch=1)

        # Graficos
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        self.canvas_fuzzy_mf = MplCanvas()
        bottom_layout.addWidget(self.canvas_fuzzy_mf)
        
        self.canvas_fuzzy_agg = MplCanvas()
        bottom_layout.addWidget(self.canvas_fuzzy_agg)
        
        main_layout.addWidget(bottom_panel, stretch=2)

        return tab

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------
    def _init_controller(self):
        self.base_controller = FuzzyController(self.config.fuzzy)
        self._log(f"Controlador difuso inicializado: {self.base_controller}")
        # Mostrar funciones de pertenencia iniciales
        self._update_mf_plot()

    def _update_config(self):
        """Actualiza configuracion desde los widgets."""
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

        self.config.metrics.comfort_min = (
            self.config.simulation.target_temperature - self.config.simulation.comfort_range
        )
        self.config.metrics.comfort_max = (
            self.config.simulation.target_temperature + self.config.simulation.comfort_range
        )

    def _on_device_changed(self, index):
        """Cambia el dispositivo controlado."""
        if index == 0:
            cfg = get_hvac_config()
            self.spin_target.setValue(22.0)
            self.spin_comfort.setValue(2.0)
            self._log("Dispositivo: HVAC (Climatizacion) seleccionado")
        else:
            cfg = get_refrigerator_config()
            self.spin_target.setValue(4.0)
            self.spin_comfort.setValue(1.5)
            self._log("Dispositivo: Refrigerador seleccionado")

        self.config.house.alpha = cfg.alpha
        self.config.house.beta = cfg.beta
        self.config.house.gamma = cfg.gamma
        self.config.house.delta = cfg.delta
        self.config.house.initial_temperature = cfg.initial_temperature
        self.config.house.hvac_max_power_kw = cfg.max_power_kw
        self.config.house.hvac_cop = cfg.cop
        self.config.house.hvac_standby_kw = cfg.standby_kw

    # ------------------------------------------------------------------
    # Inferencia Difusa Interactiva
    # ------------------------------------------------------------------
    def _on_hour_changed(self, value):
        """Disparado al mover el slider, recalcula inferencia para la hora seleccionada."""
        self.lbl_hour.setText(f"Seleccione la Hora de Simulación ({value} h):")
        if not self.base_result:
            return
        
        try:
            df = self.base_result.data
            # Encontrar la fila mas cercana a esa hora
            df_hour = df.iloc[(df['time_hours'] - value).abs().argsort()[:1]]
            if df_hour.empty:
                return
            row = df_hour.iloc[0]
            
            # Preparar inputs
            crisp_inputs = {
                'temp_error': row['temp_error'],
                'humidity': row.get('humidity', 0.5),
                'temperature_indoor': row['temperature_indoor'],
                'temperature_outdoor': row['temperature_outdoor'],
                'occupancy': row['occupancy'],
                'tariff_normalized': row['tariff_normalized'],
                'consumption_normalized': row.get('consumption_normalized', 0.5),
                'target_temperature': row.get('target_temperature', 22.0)
            }
            
            # Formatear state text
            state_text = (
                f"- T. Objetivo: {crisp_inputs['target_temperature']:.1f} C\n"
                f"- T. Interior: {crisp_inputs['temperature_indoor']:.1f} C\n"
                f"- T. Exterior: {crisp_inputs['temperature_outdoor']:.1f} C\n"
                f"- Error Temp: {crisp_inputs['temp_error']:+.2f} C\n"
                f"- Humedad: {crisp_inputs['humidity']*100:.1f} %\n"
                f"- Ocupantes: {crisp_inputs['occupancy']:.0f}\n"
                f"- Tarifa Norm: {crisp_inputs['tariff_normalized']:.2f}\n"
                f"- Consumo Norm: {crisp_inputs['consumption_normalized']:.2f}"
            )
            self.lbl_fuzzy_inputs.setText(state_text)

            # Usar el controlador apropiado
            controller = self.optimized_controller if self.optimized_controller else self.base_controller

            # Ejecutar inferencia localmente para esa hora
            output_val, detail = controller.evaluate_with_detail(crisp_inputs)
            self.lbl_fuzzy_output.setText(f"Nivel Control HVAC Exigido: {output_val:.1f} %")
            
            self._update_rules_table(detail)
            self._update_fuzzy_plots(detail)
        except Exception as e:
            self._log(f"Error evaluando hora: {e}")

    def _update_rules_table(self, detail: InferenceDetail):
        """Actualiza la tabla de reglas activadas."""
        top_rules = detail.top_rules
        self.rules_table.setRowCount(len(top_rules))

        for row, (rule, strength) in enumerate(top_rules):
            # Regla
            ant = " Y ".join(f"{v} es {s}" for v, s in rule.antecedents)
            con = rule.consequent[1]
            rule_text = f"SI {ant} ENTONCES {con}"
            self.rules_table.setItem(row, 0, QTableWidgetItem(rule_text))

            # Fuerza
            item = QTableWidgetItem(f"{strength:.3f}")
            if strength > 0.5:
                item.setForeground(QColor('#66bb6a'))
            elif strength > 0.2:
                item.setForeground(QColor('#f5a623'))
            else:
                item.setForeground(QColor('#ef5350'))
            self.rules_table.setItem(row, 1, item)

    def _update_fuzzy_plots(self, detail: InferenceDetail):
        """Actualiza los graficos de la pestana de inferencia difusa."""
        try:
            # Grafico 1: Funciones de pertenencia con valores marcados
            fig = fuzzy_plots.plot_all_membership_functions(
                self.base_controller.input_variables,
                self.base_controller.output_variable,
                detail.crisp_inputs
            )
            self.canvas_fuzzy_mf.update_figure(fig)

            # Grafico 2: Agregacion y centroide
            if detail.aggregated_output is not None:
                fig = fuzzy_plots.plot_aggregation_defuzzification(
                    self.base_controller.output_variable,
                    detail.aggregated_output,
                    detail.centroid_value
                )
                self.canvas_fuzzy_agg.update_figure(fig)

        except Exception as e:
            self._log(f"Error en graficos difusos: {e}")

    # ------------------------------------------------------------------
    # Acciones principales
    # ------------------------------------------------------------------
    def _on_run_base(self):
        self._update_config()
        self._init_controller()
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._log("Iniciando simulacion base...")
        self.statusBar().showMessage("Simulacion base en progreso...")

        self._sim_worker = SimulationWorker(self.config, self.base_controller, "base")
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.finished.connect(self._on_base_finished)
        self._sim_worker.error.connect(self._on_error)
        self._sim_worker.start()

    def _on_base_finished(self, result, metrics):
        self.base_result = result
        self.base_metrics = metrics

        self._log(f"Simulacion base completada: {result.num_steps} pasos")
        self._log(f"   Energia: {metrics.total_energy_kwh:.1f} kWh | "
                  f"Costo: ${metrics.total_cost:.2f} | "
                  f"Confort: {metrics.comfort_percentage:.1f}%")

        self._update_all_plots()
        self._update_metrics_table()

        self.slider_hour.setMaximum(self.spin_duration.value())
        self.slider_hour.setEnabled(True)
        self.slider_hour.setValue((self.spin_duration.value()) // 2)

        self.progress_bar.setValue(100)
        self.btn_run_ga.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Simulacion base completada")
        self.tabs.setCurrentIndex(0) # Poner en Dashboard

    def _on_run_ga(self):
        self._update_config()
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._log(f"Iniciando optimizacion GA: {self.config.genetic.population_size} ind x "
                  f"{self.config.genetic.num_generations} gen...")
        self.statusBar().showMessage("Optimizacion GA en progreso...")

        self._opt_worker = OptimizationWorker(self.config, self.base_controller)
        self._opt_worker.progress.connect(self._on_ga_progress)
        self._opt_worker.finished.connect(self._on_ga_finished)
        self._opt_worker.error.connect(self._on_error)
        self._opt_worker.start()

    def _on_ga_finished(self, ga_result, opt_controller):
        self.ga_result = ga_result
        self.optimized_controller = opt_controller

        self._log(f"Optimizacion completada en {ga_result.total_time_seconds:.1f}s")
        self._log(f"   Mejor fitness: {ga_result.best_fitness:.4f} | "
                  f"Evaluaciones: {ga_result.total_evaluations}")

        # Simulacion con controlador optimizado
        self._log("Ejecutando simulacion optimizada...")
        sim = Simulator(self.config)
        self.optimized_result = sim.run(
            opt_controller.get_controller_function(), label="optimizado"
        )
        self.optimized_metrics = calculate_metrics(
            self.optimized_result.data, self.config.simulation, self.config.metrics
        )

        self._log(f"   Energia opt: {self.optimized_metrics.total_energy_kwh:.1f} kWh | "
                  f"Costo: ${self.optimized_metrics.total_cost:.2f} | "
                  f"Confort: {self.optimized_metrics.comfort_percentage:.1f}%")

        # Actualizar TODOS los graficos con comparacion
        self._update_all_plots()
        self._update_ga_plot()
        self._update_metrics_table()

        # Update hourly slider with optimized controller logic
        self._on_hour_changed(self.slider_hour.value())

        self.progress_bar.setValue(100)
        self.btn_compare.setEnabled(True)
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Optimizacion GA completada")
        self.tabs.setCurrentIndex(10) # Comparacion Global

    def _on_compare(self):
        if not self.base_result or not self.optimized_result:
            self._log("Se necesitan ambas simulaciones para comparar.")
            return
        self._log("Generando comparacion global...")
        self._update_all_plots()
        self._update_metrics_table()
        self.tabs.setCurrentIndex(9)
        self._log("Comparacion generada.")

    def _on_export(self):
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
                    self.config, self.base_result, self.optimized_result,
                    self.base_metrics, self.optimized_metrics,
                    ga_hist, ga_avg, output_path=filepath
                )
                self._log(f"Reporte exportado: {filepath}")
            except Exception as e:
                self._on_error(f"Error exportando: {str(e)}")

    def _on_export_csv(self):
        dirpath = QFileDialog.getExistingDirectory(
            self, "Carpeta de exportacion", self.config.paths.outputs_dir
        )
        if dirpath:
            try:
                if self.base_result:
                    export_csv(self.base_result, dirpath)
                if self.optimized_result:
                    export_csv(self.optimized_result, dirpath)
                self._log(f"CSV exportados a: {dirpath}")
            except Exception as e:
                self._on_error(f"Error CSV: {str(e)}")

    # ------------------------------------------------------------------
    # Progreso y errores
    # ------------------------------------------------------------------
    def _on_sim_progress(self, step, total):
        self.progress_bar.setValue(int((step / max(total, 1)) * 100))

    def _on_ga_progress(self, gen, total, best_fitness):
        self.progress_bar.setValue(int((gen / max(total, 1)) * 100))
        self.statusBar().showMessage(
            f"GA: Gen {gen}/{total} | Mejor fitness: {best_fitness:.4f}"
        )

    def _on_error(self, msg):
        self._log(f"ERROR: {msg}")
        self._set_buttons_enabled(True)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Error", msg)

    def _set_buttons_enabled(self, enabled):
        self.btn_run_base.setEnabled(enabled)
        self.btn_run_ga.setEnabled(enabled and self.base_result is not None)
        self.btn_compare.setEnabled(enabled and self.optimized_result is not None)

    def _log(self, msg):
        self.log_text.append(msg)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    # ------------------------------------------------------------------
    # Actualizacion de graficos - SIEMPRE COMPARATIVOS
    # ------------------------------------------------------------------
    def _update_mf_plot(self):
        """Actualiza el grafico de funciones de pertenencia."""
        if not self.base_controller:
            return
        try:
            fig = fuzzy_plots.plot_all_membership_functions(
                self.base_controller.input_variables,
                self.base_controller.output_variable
            )
            self.canvas_mf.update_figure(fig)
        except Exception as e:
            self._log(f"Error en MF plot: {e}")

    def _update_all_plots(self):
        """Actualiza TODOS los graficos con datos actuales, siempre comparativos."""
        if not self.base_result:
            return

        df_base = self.base_result.data
        df_opt = self.optimized_result.data if self.optimized_result else None
        target = self.config.simulation.target_temperature
        comfort = self.config.simulation.comfort_range

        try:
            # Dashboard BASE (Nunca comparativo)
            fig = dashboard.create_simulation_dashboard(
                df_base, target, comfort, "Dashboard - Simulacion Base"
            )
            self.canvas_dashboard.update_figure(fig)

            # Comparacion Global Dashboard (Para tab 10)
            if df_opt is not None and self.optimized_metrics:
                fig_comp = dashboard.create_comparison_dashboard(
                    df_base, df_opt,
                    self.base_metrics, self.optimized_metrics,
                    target, comfort
                )
                self.canvas_compare.update_figure(fig_comp)

            # Temperaturas - comparativo
            fig = self._create_comparative_plot(
                df_base, df_opt, 'temperature_indoor',
                'Temperatura Interior', 'Temperatura (C)',
                target=target, comfort=comfort, show_outdoor=True
            )
            self.canvas_temp.update_figure(fig)

            # Humedad - comparativo
            if df_opt is not None and 'humidity' in df_opt.columns:
                fig = self._create_comparative_plot(
                    df_base, df_opt, 'humidity',
                    'Evolucion de Humedad Ambiental', 'Humedad (%)',
                    fill=True
                )
            else:
                fig = plots.plot_humidity(df_base)
            self.canvas_humidity.update_figure(fig)

            # HVAC - comparativo
            fig = self._create_comparative_plot(
                df_base, df_opt, 'hvac_level',
                'Nivel de Control del Dispositivo', 'Nivel (%)',
                fill=True
            )
            self.canvas_hvac.update_figure(fig)

            # Consumo - comparativo
            fig = self._create_comparative_plot(
                df_base, df_opt, 'total_consumption_kw',
                'Consumo Electrico Total', 'Consumo (kW)',
                fill=True
            )
            self.canvas_consumption.update_figure(fig)

            # Costo - comparativo
            fig = self._create_comparative_plot(
                df_base, df_opt, 'cumulative_cost',
                'Costo Acumulado', 'Costo ($)',
                fill=True
            )
            self.canvas_cost.update_figure(fig)

            # Confort - comparativo
            fig = self._create_comparative_plot(
                df_base, df_opt, 'comfort_index',
                'Indice de Confort Termico', 'Indice',
                fill=True
            )
            self.canvas_comfort.update_figure(fig)


            # MF comparison si hay optimizado
            if self.optimized_controller:
                self._update_mf_comparison_plot()

        except Exception as e:
            self._log(f"Error actualizando graficos: {e}")

    def _create_comparative_plot(self, df_base, df_opt, column,
                                  title, ylabel,
                                  target=None, comfort=None,
                                  show_outdoor=False, fill=False):
        """Crea un grafico comparativo base vs optimizado para cualquier columna."""
        import matplotlib.pyplot as plt
        plots.apply_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        time = df_base['time_hours']

        # Zona de confort
        if target is not None and comfort is not None:
            ax.axhspan(target - comfort, target + comfort,
                       alpha=0.12, color=plots.COLORS['green'], label='Zona confort')
            ax.axhline(target, color=plots.COLORS['green'],
                       linewidth=0.8, linestyle='--', alpha=0.5)

        # Temperatura exterior
        if show_outdoor and 'temperature_outdoor' in df_base.columns:
            ax.plot(time, df_base['temperature_outdoor'], color='#f5a623',
                    linewidth=2.0, alpha=0.85, linestyle='--', label='T. Exterior')

        # Base
        if fill:
            ax.fill_between(time, 0, df_base[column], alpha=0.2, color=plots.COLORS['base'])
        ax.plot(time, df_base[column], color=plots.COLORS['base'],
                linewidth=1.8, label='Base', alpha=0.85)

        # Optimizado (si existe)
        if df_opt is not None and column in df_opt.columns:
            if fill:
                ax.fill_between(time, 0, df_opt[column], alpha=0.15, color=plots.COLORS['optimized'])
            ax.plot(time, df_opt[column], color=plots.COLORS['optimized'],
                    linewidth=1.8, label='Optimizado', alpha=0.85)

        ax.set_xlabel('Tiempo (horas)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _update_ga_plot(self):
        """Actualiza graficos de optimizacion GA."""
        if not self.ga_result:
            return
        try:
            fig = plots.plot_ga_evolution(
                self.ga_result.get_fitness_history(),
                self.ga_result.get_avg_fitness_history()
            )
            self.canvas_ga.update_figure(fig)
        except Exception as e:
            self._log(f"Error en grafico GA: {e}")

    def _update_mf_comparison_plot(self):
        """Actualiza comparacion de funciones de pertenencia base vs optimizado."""
        if not self.base_controller or not self.optimized_controller:
            return
        try:
            # Mostrar comparacion para temp_error como variable principal
            fig = fuzzy_plots.plot_mf_comparison(
                'temp_error',
                self.base_controller.input_variables['temp_error'],
                self.optimized_controller.input_variables['temp_error']
            )
            self.canvas_mf.update_figure(fig)
        except Exception as e:
            self._log(f"Error en MF comparison: {e}")

    def _update_metrics_table(self):
        if not self.base_metrics:
            return
        base_dict = self.base_metrics.to_dict()
        opt_dict = self.optimized_metrics.to_dict() if self.optimized_metrics else {}

        self.metrics_table.setRowCount(len(base_dict))
        for row, (key, base_val) in enumerate(base_dict.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{base_val}"))
            opt_val = opt_dict.get(key, '-')
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{opt_val}"))
            if isinstance(opt_val, (int, float)) and isinstance(base_val, (int, float)) and base_val != 0:
                change = ((opt_val - base_val) / abs(base_val)) * 100
                item = QTableWidgetItem(f"{change:+.1f}%")
                item.setForeground(QColor('#66bb6a') if change < 0 else QColor('#ef5350'))
                self.metrics_table.setItem(row, 3, item)
            else:
                self.metrics_table.setItem(row, 3, QTableWidgetItem('-'))


# ==============================================================================
# Funcion de lanzamiento
# ==============================================================================

def launch_gui():
    """Lanza la interfaz grafica de la aplicacion."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

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
