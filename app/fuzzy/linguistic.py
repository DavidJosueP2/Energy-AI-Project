# ==============================================================================
# linguistic.py - Manejo de entradas/salidas linguisticas
# ==============================================================================
"""
Modulo para manejar el modo de entrada linguistica del sistema difuso.
Permite al usuario seleccionar etiquetas linguisticas en lugar de valores
numericos, y traduce entre ambas representaciones.

Tambien gestiona la salida dual: etiqueta dominante + valor numerico.
"""

from typing import Dict, List, Tuple, Optional
from app.fuzzy.membership import FuzzyVariable


class LinguisticInput:
    """
    Traduce etiquetas linguisticas a valores representativos
    para alimentar el motor de inferencia difusa.
    """

    def __init__(self):
        # Mapeo de etiquetas linguisticas a valores representativos
        # Cada valor es el centro de la funcion de pertenencia correspondiente
        self._mappings: Dict[str, Dict[str, float]] = {}
        self._build_default_mappings()

    def _build_default_mappings(self):
        """Construye los mapeos por defecto para todas las variables."""
        self._mappings = {
            'temperatura': {
                'baja': -4.0,
                'confortable': 0.0,
                'alta': 4.5,
                'muy_alta': 9.0,
            },
            'humedad': {
                'baja': 0.15,
                'media': 0.50,
                'alta': 0.85,
            },
            'ocupacion': {
                'vacia': 0.0,
                'baja': 1.5,
                'media': 3.0,
                'alta': 5.0,
            },
            'tarifa': {
                'barata': 0.1,
                'media': 0.45,
                'cara': 0.85,
            },
            'consumo': {
                'bajo': 0.15,
                'medio': 0.50,
                'alto': 0.85,
            },
        }

    def get_variables(self) -> List[str]:
        """Retorna la lista de variables disponibles."""
        return list(self._mappings.keys())

    def get_labels(self, variable: str) -> List[str]:
        """Retorna las etiquetas disponibles para una variable."""
        if variable not in self._mappings:
            return []
        return list(self._mappings[variable].keys())

    def to_crisp(self, variable: str, label: str) -> float:
        """
        Convierte una etiqueta linguistica a un valor crisp representativo.

        Args:
            variable: Nombre de la variable (ej: 'temperatura').
            label: Etiqueta linguistica (ej: 'alta').

        Returns:
            Valor numerico representativo del centro de la funcion de pertenencia.
        """
        if variable not in self._mappings:
            raise ValueError(f"Variable desconocida: {variable}")
        if label not in self._mappings[variable]:
            raise ValueError(f"Etiqueta '{label}' no valida para '{variable}'")
        return self._mappings[variable][label]

    def to_controller_inputs(self, selections: Dict[str, str]) -> Dict[str, float]:
        """
        Convierte un conjunto de selecciones linguisticas a
        entradas numericas para el controlador difuso.

        Args:
            selections: Dict {nombre_variable: etiqueta_seleccionada}.

        Returns:
            Dict con valores numericos listos para el controlador.
        """
        inputs = {}

        # Mapeo de nombres de GUI a nombres del controlador
        var_mapping = {
            'temperatura': 'temp_error',
            'humedad': 'humidity',
            'ocupacion': 'occupancy',
            'tarifa': 'tariff',
            'consumo': 'consumption',
        }

        for var_name, label in selections.items():
            crisp_val = self.to_crisp(var_name, label)
            controller_key = var_mapping.get(var_name, var_name)
            inputs[controller_key] = crisp_val

        return inputs

    def update_mapping(self, variable: str, label: str, value: float):
        """Actualiza el valor representativo de una etiqueta (usado por GA)."""
        if variable in self._mappings and label in self._mappings[variable]:
            self._mappings[variable][label] = value


class LinguisticOutput:
    """
    Interpreta la salida numerica del controlador difuso y la
    expresa en terminos linguisticos.
    """

    def __init__(self):
        # Rangos para clasificacion de salida [0, 100]
        self._output_ranges = {
            'muy_baja': (0, 15),
            'baja': (15, 35),
            'media': (35, 60),
            'alta': (60, 80),
            'muy_alta': (80, 100),
        }

    def classify(self, value: float) -> str:
        """
        Clasifica un valor numerico en su etiqueta linguistica dominante.

        Args:
            value: Valor de salida del controlador [0, 100].

        Returns:
            Etiqueta linguistica dominante.
        """
        value = max(0.0, min(100.0, value))

        for label, (low, high) in self._output_ranges.items():
            if low <= value <= high:
                return label

        return 'media'  # fallback

    def get_dual_output(self, value: float) -> Dict[str, object]:
        """
        Genera la salida dual completa: linguistica y numerica.

        Args:
            value: Valor numerico de salida [0, 100].

        Returns:
            Dict con valor numerico, porcentaje, etiqueta y descripcion.
        """
        label = self.classify(value)
        descriptions = {
            'muy_baja': 'El dispositivo opera al minimo o esta apagado',
            'baja': 'El dispositivo opera a potencia reducida',
            'media': 'El dispositivo opera a potencia moderada',
            'alta': 'El dispositivo opera a potencia elevada',
            'muy_alta': 'El dispositivo opera a maxima potencia',
        }

        return {
            'valor_numerico': round(value, 2),
            'porcentaje': f"{value:.1f}%",
            'etiqueta': label,
            'etiqueta_display': label.replace('_', ' ').title(),
            'descripcion': descriptions.get(label, ''),
            'valor_normalizado': round(value / 100.0, 4),
        }

    def get_all_labels(self) -> List[str]:
        """Retorna todas las etiquetas de salida disponibles."""
        return list(self._output_ranges.keys())

    def get_display_labels(self) -> List[str]:
        """Retorna etiquetas formateadas para mostrar en la GUI."""
        return [l.replace('_', ' ').title() for l in self._output_ranges.keys()]
