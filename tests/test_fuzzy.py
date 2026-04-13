# ==============================================================================
# test_fuzzy.py - Pruebas del módulo de lógica difusa
# ==============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.config import FuzzyConfig
from app.fuzzy.membership import (
    triangular_mf, trapezoidal_mf, FuzzySet, FuzzyVariable
)
from app.fuzzy.rules import create_default_rule_base
from app.fuzzy.inference import MamdaniInference
from app.fuzzy.controller import FuzzyController
from app.simulation.devices import build_device_spec


class TestMembershipFunctions:
    """Pruebas para funciones de pertenencia."""

    def test_triangular_peak(self):
        """El pico de una triangular debe ser 1.0."""
        x = np.array([5.0])
        result = triangular_mf(x, [2.0, 5.0, 8.0])
        assert abs(result[0] - 1.0) < 1e-10

    def test_triangular_zero_outside(self):
        """Fuera del triángulo el grado debe ser 0."""
        x = np.array([0.0, 10.0])
        result = triangular_mf(x, [2.0, 5.0, 8.0])
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_triangular_midpoint(self):
        """En el punto medio de la rampa, el grado debe ser ~0.5."""
        x = np.array([3.5])
        result = triangular_mf(x, [2.0, 5.0, 8.0])
        assert abs(result[0] - 0.5) < 0.01

    def test_triangular_range(self):
        """Todos los valores deben estar en [0, 1]."""
        x = np.linspace(-5, 15, 200)
        result = triangular_mf(x, [2.0, 5.0, 8.0])
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_trapezoidal_plateau(self):
        """En la meseta, el grado debe ser 1.0."""
        x = np.array([4.0, 5.0, 6.0])
        result = trapezoidal_mf(x, [2.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_left_shoulder(self):
        """Función con hombro izquierdo (a == b)."""
        x = np.array([0.0, 1.0, 3.0, 5.0])
        result = triangular_mf(x, [2.0, 2.0, 5.0])
        assert result[0] == 1.0  # Antes de a
        assert result[1] == 1.0  # Antes de b
        assert result[3] == 0.0  # Después de c

    def test_trapezoidal_right_shoulder(self):
        """Función trapezoidal con hombro derecho explícito."""
        x = np.array([70.0, 90.0, 100.0, 110.0])
        result = trapezoidal_mf(x, [82.0, 100.0, 100.0, 100.0])
        assert result[0] == 0.0
        assert result[1] > 0.0
        assert result[2] == 1.0
        assert result[3] == 1.0


class TestFuzzyVariable:
    """Pruebas para variables difusas."""

    def test_create_variable(self):
        """Crear una variable con conjuntos difusos."""
        var = FuzzyVariable('test', (0, 10), 100)
        var.add_set('bajo', 'triangular', [0, 0, 5])
        var.add_set('alto', 'triangular', [5, 10, 10])
        assert len(var.sets) == 2

    def test_fuzzify(self):
        """Fuzzificar un valor debe retornar grados para todos los conjuntos."""
        var = FuzzyVariable('test', (0, 10), 100)
        var.add_set('bajo', 'triangular', [0, 0, 5])
        var.add_set('alto', 'triangular', [5, 10, 10])
        
        result = var.fuzzify(2.5)
        assert 'bajo' in result
        assert 'alto' in result
        assert result['bajo'] > 0
        assert result['alto'] == 0

    def test_validate(self):
        """Variable con buena cobertura debe superar validación."""
        var = FuzzyVariable('test', (0, 10), 100)
        var.add_set('bajo', 'triangular', [0, 0, 5])
        var.add_set('medio', 'triangular', [3, 5, 7])
        var.add_set('alto', 'triangular', [5, 10, 10])
        assert var.validate()

    def test_mixed_triangular_and_trapezoidal_sets(self):
        """Una variable puede mezclar triángulos internos con hombros trapezoidales."""
        var = FuzzyVariable('test', (0, 10), 100)
        var.add_set('bajo', 'trapezoidal', [0, 0, 0, 3])
        var.add_set('medio', 'triangular', [2, 5, 8])
        var.add_set('alto', 'trapezoidal', [7, 10, 10, 10])
        assert var.validate()
        result = var.fuzzify(1.0)
        assert result['bajo'] > 0.0
        assert result['alto'] == 0.0


class TestRuleSet:
    """Pruebas para la base de reglas."""

    def test_default_rules_exist(self):
        """La base por defecto debe tener reglas suficientes."""
        rb = create_default_rule_base()
        assert rb.num_rules >= 20, f"Solo {rb.num_rules} reglas, se esperan ≥20"

    def test_rules_well_formed(self):
        """Todas las reglas deben tener antecedentes y consecuente."""
        rb = create_default_rule_base()
        for rule in rb.get_rules():
            assert len(rule.antecedents) > 0
            assert len(rule.consequent) == 2
            assert 0 < rule.weight <= 1.0


class TestFuzzyController:
    """Pruebas para el controlador difuso completo."""

    def setup_method(self):
        self.controller = FuzzyController()

    def test_evaluate_returns_number(self):
        """La evaluación debe retornar un número."""
        result = self.controller.evaluate({
            'temp_error': 3.0,
            'occupancy': 2.0,
            'tariff_normalized': 0.5,
            'consumption_normalized': 0.3,
        })
        assert isinstance(result, float)

    def test_output_range(self):
        """La salida debe estar en [0, 100]."""
        for te in [-5, 0, 3, 7, 12]:
            for occ in [0, 2, 4]:
                result = self.controller.evaluate({
                    'temp_error': te,
                    'occupancy': occ,
                    'tariff_normalized': 0.5,
                    'consumption_normalized': 0.5,
                })
                assert 0 <= result <= 100, f"Salida fuera de rango: {result}"

    def test_signed_error_demands_more_than_comfort(self):
        """Errores frios o calientes deben exigir mas control que una condicion confortable."""
        hot = self.controller.evaluate({
            'temp_error': 8.0, 'occupancy': 3.0, 'humidity': 0.5,
            'tariff_normalized': 0.3, 'consumption_normalized': 0.3,
        })
        cold = self.controller.evaluate({
            'temp_error': -3.0, 'occupancy': 3.0, 'humidity': 0.5,
            'tariff_normalized': 0.3, 'consumption_normalized': 0.3,
        })
        comfort = self.controller.evaluate({
            'temp_error': 0.0, 'occupancy': 3.0, 'humidity': 0.5,
            'tariff_normalized': 0.3, 'consumption_normalized': 0.3,
        })
        assert hot > comfort, f"Hot ({hot}) should be > Comfort ({comfort})"
        assert cold > comfort, f"Cold ({cold}) should be > Comfort ({comfort})"

    def test_clone(self):
        """Clonar debe crear una copia independiente."""
        clone = self.controller.clone()
        original_result = self.controller.evaluate({
            'temp_error': 5.0, 'occupancy': 2.0,
            'tariff_normalized': 0.5, 'consumption_normalized': 0.5,
        })
        clone_result = clone.evaluate({
            'temp_error': 5.0, 'occupancy': 2.0,
            'tariff_normalized': 0.5, 'consumption_normalized': 0.5,
        })
        assert abs(original_result - clone_result) < 1e-10

    def test_narrower_comfort_range_increases_hvac_response(self):
        """Un rango de confort mas estrecho debe endurecer la respuesta ante el mismo error."""
        wide = self.controller.evaluate({
            'temp_error': 1.0,
            'occupancy': 2.0,
            'humidity': 0.5,
            'tariff_normalized': 0.5,
            'comfort_range': 3.5,
        })
        narrow = self.controller.evaluate({
            'temp_error': 1.0,
            'occupancy': 2.0,
            'humidity': 0.5,
            'tariff_normalized': 0.5,
            'comfort_range': 0.5,
        })
        assert narrow > wide, f"Rango estrecho ({narrow}) deberia exigir mas control que rango amplio ({wide})"

    def test_get_set_params(self):
        """Debe poder obtener y establecer parámetros."""
        params = self.controller.get_membership_params()
        assert 'temp_error' in params
        assert 'hvac_output' in params
        
        # Modificar y re-establecer
        self.controller.set_membership_params(params)

    def test_device_specs_use_explicit_trapezoidal_extremes(self):
        """Los extremos del baseline deben quedar declarados como trapezoidales explícitos."""
        hvac = build_device_spec("hvac")
        refri = build_device_spec("refrigerador")

        assert hvac.get_variable("temp_error").get_mf_type("baja") == "trapezoidal"
        assert hvac.get_variable("tariff").get_mf_type("cara") == "trapezoidal"
        assert hvac.output_variable.get_mf_type("muy_baja") == "trapezoidal"
        assert len(hvac.get_variable("temp_error").sets["baja"]) == 4
        assert len(hvac.output_variable.sets["muy_alta"]) == 4

        assert refri.get_variable("device_temperature").get_mf_type("muy_alta") == "trapezoidal"
        assert refri.get_variable("door_openings").get_mf_type("baja") == "trapezoidal"
        assert refri.output_variable.get_mf_type("muy_alta") == "trapezoidal"
        assert len(refri.get_variable("load_level").sets["alta"]) == 4
        assert len(refri.output_variable.sets["muy_baja"]) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
