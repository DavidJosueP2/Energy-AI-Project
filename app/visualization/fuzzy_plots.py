import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Tuple, Optional

from app.fuzzy.membership import FuzzyVariable, FuzzySet
from app.fuzzy.rules import FuzzyRule
from app.visualization.plots import apply_style, COLORS


def plot_membership_functions(variable: FuzzyVariable,
                               current_value: Optional[float] = None,
                               title_prefix: str = "") -> Figure:
    """
    Grafica las funciones de pertenencia de una variable difusa.

    Si se proporciona current_value, marca el valor y muestra
    los grados de pertenencia activados.

    Args:
        variable: Variable difusa a graficar.
        current_value: Valor actual para mostrar activacion.
        title_prefix: Prefijo para el titulo del grafico.

    Returns:
        Figura matplotlib con las funciones de pertenencia.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623',
                   '#ab47bc', '#ff7043', '#4fc3f7', '#ef5350']
    universe = variable.universe

    for idx, (set_name, fuzzy_set) in enumerate(variable.sets.items()):
        color = colors_list[idx % len(colors_list)]
        membership = fuzzy_set.evaluate(universe)
        ax.plot(universe, membership, color=color, linewidth=2,
                label=set_name.replace('_', ' ').title())
        ax.fill_between(universe, 0, membership, alpha=0.08, color=color)
    
    if current_value is not None:
        ax.axvline(current_value, color='white', linewidth=2,
                   linestyle='--', alpha=0.8, label=f'Valor: {current_value:.2f}')

        # Mostrar grados de pertenencia
        degrees = variable.fuzzify(current_value)
        y_offset = 0.95
        for set_name, degree in degrees.items():
            if degree > 0.01:
                ax.annotate(
                    f'{set_name}: {degree:.3f}',
                    xy=(current_value, degree),
                    xytext=(10, y_offset * 30),
                    textcoords='offset points',
                    fontsize=9, color='#eee',
                    arrowprops=dict(arrowstyle='->', color='#888', lw=0.8),
                )
                y_offset -= 0.15

    title = f"{title_prefix}{variable.name.replace('_', ' ').title()}"
    ax.set_title(f"Funciones de Pertenencia: {title}",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Universo de Discurso')
    ax.set_ylabel('Grado de Pertenencia')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_all_membership_functions(variables: Dict[str, FuzzyVariable],
                                   output_variable: FuzzyVariable,
                                   current_inputs: Optional[Dict[str, float]] = None
                                   ) -> Figure:
    """
    Grafica las funciones de pertenencia de todas las variables
    en una sola figura con subplots.
    """
    apply_style()
    all_vars = list(variables.items()) + [('Salida', output_variable)]
    n_vars = len(all_vars)
    cols = 2
    rows = (n_vars + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623',
                   '#ab47bc', '#ff7043', '#4fc3f7', '#ef5350']

    for idx, (var_name, variable) in enumerate(all_vars):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        universe = variable.universe

        for s_idx, (set_name, fuzzy_set) in enumerate(variable.sets.items()):
            color = colors_list[s_idx % len(colors_list)]
            membership = fuzzy_set.evaluate(universe)
            ax.plot(universe, membership, color=color, linewidth=1.8,
                    label=set_name.replace('_', ' ').title())
            ax.fill_between(universe, 0, membership, alpha=0.06, color=color)

        if current_inputs and var_name in current_inputs:
            val = current_inputs[var_name]
            ax.axvline(val, color='white', linewidth=1.5,
                       linestyle='--', alpha=0.7)

        ax.set_title(var_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    for idx in range(len(all_vars), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Funciones de Pertenencia del Sistema Difuso',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig


def plot_rule_activations(rules_with_strength: List[Tuple[FuzzyRule, float]],
                           top_n: int = 15) -> Figure:
    """
    Grafica las reglas activadas con sus grados de disparo.

    Args:
        rules_with_strength: Lista de (regla, fuerza_de_disparo).
        top_n: Numero maximo de reglas a mostrar.

    Returns:
        Figura con barras horizontales de activacion.
    """
    apply_style()

    # Filtrar reglas con activacion > 0 y ordenar
    active = [(r, s) for r, s in rules_with_strength if s > 0.01]
    active.sort(key=lambda x: x[1], reverse=True)
    active = active[:top_n]

    if not active:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'Ninguna regla activada',
                ha='center', va='center', fontsize=14, color='#888')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    fig, ax = plt.subplots(figsize=(12, max(3, len(active) * 0.45)))

    labels = []
    strengths = []
    for rule, strength in active:
        # Construir etiqueta legible
        ant = " Y ".join(f"{v} es {s}" for v, s in rule.antecedents)
        con = f"{rule.consequent[1]}"
        label = f"SI {ant} => {con}"
        if len(label) > 80:
            label = label[:77] + "..."
        labels.append(label)
        strengths.append(strength)

    y_pos = np.arange(len(labels))
    colors = [plt.cm.RdYlGn(s) for s in strengths]

    bars = ax.barh(y_pos, strengths, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Grado de Activacion', fontsize=11)
    ax.set_title('Reglas Difusas Activadas', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, strengths):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8, color='#ccc')

    fig.tight_layout()
    return fig


def plot_aggregation_defuzzification(output_variable: FuzzyVariable,
                                      aggregated: np.ndarray,
                                      centroid_value: float) -> Figure:
    """
    Grafica el proceso de agregacion y desfuzzificacion.

    Muestra:
    - Funcion agregada (area sombreada)
    - Linea del centroide
    - Valor defuzzificado

    Args:
        output_variable: Variable de salida.
        aggregated: Array con la funcion de pertenencia agregada.
        centroid_value: Valor del centroide calculado.

    Returns:
        Figura con la visualizacion de la desfuzzificacion.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    universe = output_variable.universe

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623', '#ab47bc']
    for idx, (set_name, fuzzy_set) in enumerate(output_variable.sets.items()):
        color = colors_list[idx % len(colors_list)]
        membership = fuzzy_set.evaluate(universe)
        ax.plot(universe, membership, color=color, linewidth=1,
                alpha=0.3, linestyle='--')

    ax.fill_between(universe, 0, aggregated, alpha=0.5, color='#53d8fb',
                    label='Funcion agregada')
    ax.plot(universe, aggregated, color='#53d8fb', linewidth=2.5)

    ax.axvline(centroid_value, color='#e94560', linewidth=2.5,
               linestyle='-', label=f'Centroide: {centroid_value:.2f}')

    max_agg = max(aggregated) if max(aggregated) > 0 else 0.5
    ax.annotate(
        f'Salida = {centroid_value:.2f}',
        xy=(centroid_value, 0),
        xytext=(centroid_value + (output_variable.universe_range[1] - centroid_value) * 0.3, max_agg * 0.8),
        fontsize=12, fontweight='bold', color='#e94560',
        arrowprops=dict(arrowstyle='->', color='#e94560', lw=2),
    )

    ax.set_title('Agregacion y Desfuzzificacion (Centroide)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Nivel de Control')
    ax.set_ylabel('Grado de Pertenencia')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_inference_summary(variables: Dict[str, FuzzyVariable],
                            output_variable: FuzzyVariable,
                            crisp_inputs: Dict[str, float],
                            aggregated: np.ndarray,
                            centroid_value: float,
                            rules_with_strength: List[Tuple] = None) -> Figure:
    """
    Grafica un resumen completo del proceso de inferencia en una sola figura.

    Layout:
    - Fila 1: Funciones de pertenencia de entradas con valores marcados
    - Fila 2: Reglas activadas (izquierda) + Agregacion/Centroide (derecha)
    """
    apply_style()
    n_inputs = len(variables)
    fig = plt.figure(figsize=(16, 10))

    gs_top = GridSpec(1, n_inputs, figure=fig,
                      left=0.05, right=0.95, top=0.95, bottom=0.55,
                      wspace=0.3)

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623',
                   '#ab47bc', '#ff7043']

    for col, (var_name, variable) in enumerate(variables.items()):
        ax = fig.add_subplot(gs_top[0, col])
        universe = variable.universe

        for s_idx, (set_name, fuzzy_set) in enumerate(variable.sets.items()):
            color = colors_list[s_idx % len(colors_list)]
            membership = fuzzy_set.evaluate(universe)
            ax.plot(universe, membership, color=color, linewidth=1.5,
                    label=set_name.replace('_', ' '))
            ax.fill_between(universe, 0, membership, alpha=0.05, color=color)

        if var_name in crisp_inputs:
            val = crisp_inputs[var_name]
            ax.axvline(val, color='white', linewidth=1.5, linestyle='--', alpha=0.8)

        ax.set_title(var_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)

    gs_bot = GridSpec(1, 2, figure=fig,
                      left=0.05, right=0.95, top=0.48, bottom=0.05,
                      wspace=0.3)

    ax_rules = fig.add_subplot(gs_bot[0, 0])
    if rules_with_strength:
        active = [(r, s) for r, s in rules_with_strength if s > 0.01]
        active.sort(key=lambda x: x[1], reverse=True)
        active = active[:8]

        if active:
            labels = []
            strengths = []
            for rule, s in active:
                ant = " Y ".join(f"{v}={sv}" for v, sv in rule.antecedents)
                label = f"{ant} => {rule.consequent[1]}"
                if len(label) > 50:
                    label = label[:47] + "..."
                labels.append(label)
                strengths.append(s)

            y_pos = np.arange(len(labels))
            bar_colors = [plt.cm.RdYlGn(s) for s in strengths]
            ax_rules.barh(y_pos, strengths, color=bar_colors, height=0.6)
            ax_rules.set_yticks(y_pos)
            ax_rules.set_yticklabels(labels, fontsize=7)
            ax_rules.set_xlim(0, 1.05)
            ax_rules.invert_yaxis()
        else:
            ax_rules.text(0.5, 0.5, 'Sin reglas activas',
                         ha='center', va='center', color='#888')
    else:
        ax_rules.text(0.5, 0.5, 'Sin datos de reglas',
                     ha='center', va='center', color='#888')

    ax_rules.set_title('Reglas Activadas (Top)', fontsize=10, fontweight='bold')
    ax_rules.grid(True, axis='x', alpha=0.2)

    ax_agg = fig.add_subplot(gs_bot[0, 1])
    universe = output_variable.universe

    for s_idx, (set_name, fuzzy_set) in enumerate(output_variable.sets.items()):
        color = colors_list[s_idx % len(colors_list)]
        membership = fuzzy_set.evaluate(universe)
        ax_agg.plot(universe, membership, color=color, linewidth=0.8,
                    alpha=0.3, linestyle='--')

    ax_agg.fill_between(universe, 0, aggregated, alpha=0.5, color='#53d8fb')
    ax_agg.plot(universe, aggregated, color='#53d8fb', linewidth=2)
    ax_agg.axvline(centroid_value, color='#e94560', linewidth=2.5,
                   label=f'Centroide: {centroid_value:.2f}')

    ax_agg.set_title('Agregacion + Desfuzzificacion', fontsize=10, fontweight='bold')
    ax_agg.set_ylim(-0.05, 1.15)
    ax_agg.legend(fontsize=9)
    ax_agg.grid(True, alpha=0.2)

    fig.suptitle('Proceso Completo de Inferencia Difusa',
                 fontsize=14, fontweight='bold', y=0.99)

    return fig


def plot_mf_comparison(var_name: str,
                        base_variable: FuzzyVariable,
                        optimized_variable: FuzzyVariable) -> Figure:
    """
    Compara funciones de pertenencia antes y despues de la optimizacion GA.

    Args:
        var_name: Nombre de la variable.
        base_variable: Variable con parametros originales.
        optimized_variable: Variable con parametros optimizados.

    Returns:
        Figura con la comparacion lado a lado.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623',
                   '#ab47bc', '#ff7043']

    for idx, (set_name, fuzzy_set) in enumerate(base_variable.sets.items()):
        color = colors_list[idx % len(colors_list)]
        membership = fuzzy_set.evaluate(base_variable.universe)
        ax1.plot(base_variable.universe, membership, color=color,
                 linewidth=2, label=set_name.replace('_', ' '))
        ax1.fill_between(base_variable.universe, 0, membership,
                        alpha=0.08, color=color)

    ax1.set_title('Funciones Base (Original)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Universo de Discurso')
    ax1.set_ylabel('Grado de Pertenencia')
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    for idx, (set_name, fuzzy_set) in enumerate(optimized_variable.sets.items()):
        color = colors_list[idx % len(colors_list)]
        membership = fuzzy_set.evaluate(optimized_variable.universe)
        ax2.plot(optimized_variable.universe, membership, color=color,
                 linewidth=2, label=set_name.replace('_', ' '))
        ax2.fill_between(optimized_variable.universe, 0, membership,
                        alpha=0.08, color=color)

    ax2.set_title('Funciones Optimizadas (GA)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Universo de Discurso')
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f'Comparacion de Funciones de Pertenencia: {var_name.replace("_", " ").title()}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_all_mf_comparisons(
    base_variables: Dict[str, FuzzyVariable],
    optimized_variables: Dict[str, FuzzyVariable],
    base_output_variable: FuzzyVariable,
    optimized_output_variable: FuzzyVariable,
) -> Figure:
    """
    Compara todas las funciones de pertenencia optimizadas vs originales.

    Se construye una grilla con dos columnas:
    - izquierda: base/original
    - derecha: optimizado

    Cada fila corresponde a una variable de entrada o a la salida.
    """
    apply_style()

    variable_names = list(base_variables.keys()) + ["__output__"]
    rows = len(variable_names)
    fig, axes = plt.subplots(rows, 2, figsize=(16, max(7, rows * 4.1)), sharey=False)

    if rows == 1:
        axes = np.array([axes])

    colors_list = ['#e94560', '#53d8fb', '#66bb6a', '#f5a623',
                   '#ab47bc', '#ff7043', '#4fc3f7', '#ef5350']

    for row_idx, variable_name in enumerate(variable_names):
        if variable_name == "__output__":
            display_name = "Salida de Control"
            base_variable = base_output_variable
            optimized_variable = optimized_output_variable
        else:
            display_name = variable_name.replace('_', ' ').title()
            base_variable = base_variables[variable_name]
            optimized_variable = optimized_variables[variable_name]

        for col_idx, (ax, variable, subtitle) in enumerate([
            (axes[row_idx][0], base_variable, "Base (Original)"),
            (axes[row_idx][1], optimized_variable, "Optimizado (GA)"),
        ]):
            for idx, (set_name, fuzzy_set) in enumerate(variable.sets.items()):
                color = colors_list[idx % len(colors_list)]
                membership = fuzzy_set.evaluate(variable.universe)
                ax.plot(variable.universe, membership, color=color, linewidth=2,
                        label=set_name.replace('_', ' '))
                ax.fill_between(variable.universe, 0, membership, alpha=0.08, color=color)

            ax.set_ylim(-0.05, 1.15)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlabel('Universo de Discurso')
            if col_idx == 0:
                ax.set_ylabel('Grado de Pertenencia')
            ax.set_title(f"{display_name} | {subtitle}", fontsize=10.5, fontweight='bold')

    fig.suptitle('Comparacion Completa de Funciones de Pertenencia', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975], h_pad=2.2, w_pad=1.2)
    return fig
