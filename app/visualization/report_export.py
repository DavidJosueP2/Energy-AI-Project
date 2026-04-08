# ==============================================================================
# report_export.py - Exportación de reportes
# ==============================================================================
"""
Genera reportes en múltiples formatos:
- CSV con datos de simulación
- Imágenes PNG de gráficos
- Reporte HTML profesional con métricas, gráficos y resumen
"""

import os
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from app.config import AppConfig
from app.simulation.simulator import SimulationResult
from app.simulation.metrics import PerformanceMetrics
from app.visualization import plots


def export_csv(result: SimulationResult, output_dir: str, prefix: str = "sim") -> str:
    """Exporta datos de simulación a CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_{result.label}_data.csv"
    filepath = os.path.join(output_dir, filename)
    result.to_csv(filepath)
    return filepath


def export_plots_png(result: SimulationResult,
                      output_dir: str,
                      config: AppConfig,
                      prefix: str = "sim") -> List[str]:
    """Exporta todos los gráficos como imágenes PNG."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    df = result.data
    target = config.simulation.target_temperature
    comfort = config.simulation.comfort_range
    
    plot_functions = [
        ('temperaturas', lambda: plots.plot_temperatures(df, target, comfort)),
        ('hvac', lambda: plots.plot_hvac_level(df)),
        ('consumo', lambda: plots.plot_consumption(df)),
        ('costo', lambda: plots.plot_cumulative_cost(df)),
        ('confort', lambda: plots.plot_comfort(df, target)),
        ('ocupacion_tarifa', lambda: plots.plot_occupancy_tariff(df)),
        ('distribucion_consumo', lambda: plots.plot_consumption_distribution(df)),
    ]
    
    for name, plot_fn in plot_functions:
        try:
            fig = plot_fn()
            filepath = os.path.join(output_dir, f"{prefix}_{result.label}_{name}.png")
            plots.save_figure(fig, filepath)
            saved.append(filepath)
        except Exception as e:
            print(f"Error generando gráfico {name}: {e}")
    
    return saved


def _fig_to_base64(fig) -> str:
    """Convierte una figura matplotlib a base64 para incrustar en HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(config: AppConfig,
                          base_result: Optional[SimulationResult],
                          opt_result: Optional[SimulationResult],
                          base_metrics: Optional[PerformanceMetrics],
                          opt_metrics: Optional[PerformanceMetrics],
                          ga_fitness_history: Optional[List[float]] = None,
                          ga_avg_history: Optional[List[float]] = None,
                          output_path: Optional[str] = None) -> str:
    """
    Genera un reporte HTML profesional con resultados completos.
    
    Returns:
        String HTML del reporte (también guardado en disco si output_path se especifica).
    """
    # Generar gráficos como base64
    images = {}
    target = config.simulation.target_temperature
    comfort = config.simulation.comfort_range
    
    if base_result is not None:
        df = base_result.data
        images['temp_base'] = _fig_to_base64(plots.plot_temperatures(df, target, comfort))
        images['hvac_base'] = _fig_to_base64(plots.plot_hvac_level(df))
        images['consumo_base'] = _fig_to_base64(plots.plot_consumption(df))
        images['costo_base'] = _fig_to_base64(plots.plot_cumulative_cost(df))
        images['confort_base'] = _fig_to_base64(plots.plot_comfort(df, target))
        images['dist_base'] = _fig_to_base64(plots.plot_consumption_distribution(df))
    
    if ga_fitness_history:
        images['ga_evolution'] = _fig_to_base64(
            plots.plot_ga_evolution(ga_fitness_history, ga_avg_history))
    
    if base_result and opt_result:
        images['comparison'] = _fig_to_base64(
            plots.plot_comparison(base_result.data, opt_result.data, target, comfort))
    
    if base_metrics and opt_metrics:
        images['metrics_bars'] = _fig_to_base64(
            plots.plot_metrics_comparison_bars(base_metrics, opt_metrics))
    
    # Construir HTML
    html = _build_html(config, base_metrics, opt_metrics, images)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    return html


def _build_html(config: AppConfig,
                base_metrics: Optional[PerformanceMetrics],
                opt_metrics: Optional[PerformanceMetrics],
                images: Dict[str, str]) -> str:
    """Construye el HTML del reporte."""
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Tabla de métricas
    metrics_html = ""
    if base_metrics:
        base_dict = base_metrics.to_dict()
        opt_dict = opt_metrics.to_dict() if opt_metrics else {}
        
        rows = ""
        for key, val in base_dict.items():
            opt_val = opt_dict.get(key, '-')
            if isinstance(opt_val, (int, float)) and isinstance(val, (int, float)) and val != 0:
                change = ((opt_val - val) / abs(val)) * 100
                change_str = f'<span style="color:{"#66bb6a" if change <= 0 else "#ef5350"}">{change:+.1f}%</span>'
            else:
                change_str = '-'
            
            rows += f"""
            <tr>
                <td>{key}</td>
                <td>{val}</td>
                <td>{opt_val if opt_val != '-' else '-'}</td>
                <td>{change_str}</td>
            </tr>"""
        
        metrics_html = f"""
        <h2>📊 Métricas de Desempeño</h2>
        <table>
            <thead>
                <tr><th>Métrica</th><th>Base</th><th>Optimizado</th><th>Cambio</th></tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""
    
    # Imágenes
    images_html = ""
    img_titles = {
        'temp_base': '🌡️ Evolución de Temperaturas',
        'hvac_base': '❄️ Control HVAC',
        'consumo_base': '⚡ Consumo Eléctrico',
        'costo_base': '💰 Costo Acumulado',
        'confort_base': '😊 Confort Térmico',
        'dist_base': '📈 Distribución de Consumo',
        'ga_evolution': '🧬 Evolución del Algoritmo Genético',
        'comparison': '🔄 Comparación Base vs Optimizado',
        'metrics_bars': '📊 Métricas Comparativas',
    }
    
    for key, title in img_titles.items():
        if key in images:
            images_html += f"""
            <div class="chart-section">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{images[key]}" alt="{title}">
            </div>"""
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte - Sistema de Gestión Energética Inteligente</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            padding: 40px;
            line-height: 1.6;
        }}
        h1 {{
            text-align: center;
            font-size: 28px;
            color: #e94560;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        h2 {{
            color: #53d8fb;
            margin: 30px 0 15px;
            border-bottom: 2px solid #53d8fb33;
            padding-bottom: 5px;
        }}
        h3 {{ color: #f5a623; margin: 20px 0 10px; }}
        .config-box {{
            background: #16213e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .config-box p {{ margin: 4px 0; font-size: 14px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #0f3460;
            color: #53d8fb;
            padding: 12px 15px;
            text-align: left;
            font-size: 13px;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #222;
            font-size: 13px;
        }}
        tr:hover {{ background: #1a1a3e; }}
        .chart-section {{
            margin: 25px 0;
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #333;
        }}
        .chart-section img {{
            width: 100%;
            border-radius: 4px;
            margin-top: 10px;
        }}
        .footer {{
            text-align: center;
            color: #555;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>🏠 Sistema Inteligente de Gestión Energética Residencial</h1>
    <p class="subtitle">Reporte generado el {now}</p>

    <h2>⚙️ Configuración del Experimento</h2>
    <div class="config-box">
        <p><strong>Duración:</strong> {config.simulation.horizon_hours} horas</p>
        <p><strong>Escenario:</strong> {config.simulation.scenario_type}</p>
        <p><strong>Temperatura objetivo:</strong> {config.simulation.target_temperature}°C ± {config.simulation.comfort_range}°C</p>
        <p><strong>Semilla:</strong> {config.simulation.random_seed}</p>
        <p><strong>Paso temporal:</strong> {config.simulation.time_step_hours} horas</p>
        <p><strong>Población GA:</strong> {config.genetic.population_size} individuos</p>
        <p><strong>Generaciones GA:</strong> {config.genetic.num_generations}</p>
    </div>

    {metrics_html}

    <h2>📈 Gráficos de Resultados</h2>
    {images_html}

    <div class="footer">
        <p>Proyecto de Inteligencia Artificial — Gestión Energética Residencial con Lógica Difusa y Algoritmo Genético</p>
    </div>
</body>
</html>"""
    
    return html
