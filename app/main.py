# ==============================================================================
# main.py - Punto de entrada principal de la aplicación
# ==============================================================================
"""
Coordina la ejecución del sistema completo.
Puede ejecutarse en modo GUI o en modo CLI para simulaciones headless.
"""

import sys
import os
import argparse

# Asegurar que el directorio raíz del proyecto esté en el path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Sistema Inteligente de Gestión Energética Residencial"
    )
    parser.add_argument('--cli', action='store_true',
                        help='Ejecutar en modo línea de comandos (sin GUI)')
    parser.add_argument('--scenario', type=str, default='verano',
                        choices=['verano', 'invierno', 'primavera', 'mixto'],
                        help='Tipo de escenario (default: verano)')
    parser.add_argument('--hours', type=int, default=72,
                        help='Duración en horas (default: 72)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')
    parser.add_argument('--optimize', action='store_true',
                        help='Ejecutar optimización genética')
    parser.add_argument('--export', type=str, default=None,
                        help='Ruta para exportar reporte HTML')
    
    args = parser.parse_args()
    
    if args.cli:
        run_cli(args)
    else:
        run_gui()


def run_gui():
    """Lanza la interfaz gráfica."""
    from app.ui.gui import launch_gui
    launch_gui()


def run_cli(args):
    """Ejecuta en modo CLI."""
    from app.config import AppConfig
    from app.fuzzy.controller import FuzzyController
    from app.simulation.simulator import Simulator
    from app.simulation.metrics import calculate_metrics, compare_metrics
    from app.genetic.optimizer import GeneticOptimizer
    from app.visualization.report_export import generate_html_report, export_csv
    
    print("=" * 60)
    print("  Sistema Inteligente de Gestión Energética Residencial")
    print("  Modo: Línea de Comandos")
    print("=" * 60)
    
    # Configurar
    config = AppConfig()
    config.simulation.horizon_hours = args.hours
    config.simulation.random_seed = args.seed
    config.simulation.scenario_type = args.scenario
    config.metrics.comfort_min = config.simulation.target_temperature - config.simulation.comfort_range
    config.metrics.comfort_max = config.simulation.target_temperature + config.simulation.comfort_range
    
    print(f"\nEscenario: {args.scenario} | Duración: {args.hours}h | Semilla: {args.seed}")
    
    # Controlador base
    controller = FuzzyController(config.fuzzy)
    print(f"Controlador: {controller}")
    
    # Simulación base
    print("\n--- Ejecutando simulación base ---")
    simulator = Simulator(config)
    base_result = simulator.run(
        controller.get_controller_function(),
        label="base",
        progress_callback=lambda s, t: print(f"\r  Progreso: {s}/{t}", end='')
    )
    print()
    
    base_metrics = calculate_metrics(
        base_result.data, config.simulation, config.metrics
    )
    print("\nMétricas Base:")
    for k, v in base_metrics.to_dict().items():
        print(f"  {k}: {v}")
    
    # Optimización (opcional)
    opt_result = None
    opt_metrics = None
    ga_result = None
    opt_controller = None
    
    if args.optimize:
        print("\n--- Ejecutando optimización genética ---")
        optimizer = GeneticOptimizer(config, controller)
        ga_result = optimizer.optimize(
            progress_callback=lambda g, t, f: print(
                f"\r  Generación {g}/{t} | Fitness: {f:.4f}", end=''
            )
        )
        print()
        
        opt_controller = optimizer.decode_best(ga_result)
        print(f"Mejor fitness: {ga_result.best_fitness:.4f}")
        print(f"Tiempo total: {ga_result.total_time_seconds:.1f}s")
        
        # Simulación optimizada
        print("\n--- Ejecutando simulación optimizada ---")
        opt_sim_result = simulator.run(
            opt_controller.get_controller_function(), label="optimizado"
        )
        opt_result = opt_sim_result
        opt_metrics = calculate_metrics(
            opt_result.data, config.simulation, config.metrics
        )
        
        print("\nMétricas Optimizado:")
        for k, v in opt_metrics.to_dict().items():
            print(f"  {k}: {v}")
        
        # Comparación
        print("\n--- Comparación ---")
        comparison = compare_metrics(base_metrics, opt_metrics)
        for k, v in comparison.items():
            change = v['cambio_%']
            arrow = "+" if change > 0 else "-" if change < 0 else "="
            print(f"  {k}: {v['base']} -> {v['optimizado']} ({arrow}{abs(change):.1f}%)")
    
    # Exportar
    export_path = args.export
    if not export_path:
        export_path = os.path.join(config.paths.outputs_dir, "reporte.html")
    
    ga_hist = ga_result.get_fitness_history() if ga_result else None
    ga_avg = ga_result.get_avg_fitness_history() if ga_result else None
    
    generate_html_report(
        config, base_result, opt_result,
        base_metrics, opt_metrics,
        ga_hist, ga_avg,
        output_path=export_path
    )
    print(f"\n[OK] Reporte HTML generado: {export_path}")
    
    # CSV
    csv_dir = config.paths.outputs_dir
    export_csv(base_result, csv_dir)
    if opt_result:
        export_csv(opt_result, csv_dir)
    print(f"[OK] Datos CSV exportados a: {csv_dir}")
    
    print("\n" + "=" * 60)
    print("  Ejecución completada exitosamente.")
    print("=" * 60)


if __name__ == '__main__':
    main()
