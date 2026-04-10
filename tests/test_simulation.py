# ==============================================================================
# test_simulation.py - Pruebas del módulo de simulación
# ==============================================================================
"""
Pruebas automáticas para:
- Generación de escenarios y perfiles ambientales
- Modelo térmico de la vivienda
- Motor de simulación completo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.config import AppConfig, SimulationConfig, EnvironmentConfig
from app.simulation.environment import EnvironmentProfile
from app.simulation.house_model import HouseModel
from app.simulation.simulator import Simulator, SimulationResult
from app.simulation.scenario_generator import get_scenario_configs, AVAILABLE_SCENARIOS
from app.simulation.metrics import calculate_metrics


class TestEnvironmentProfile:
    """Pruebas para la generación de perfiles ambientales."""

    def setup_method(self):
        sim_config = SimulationConfig(horizon_hours=24, random_seed=42)
        env_config = EnvironmentConfig()
        self.profile = EnvironmentProfile(sim_config, env_config)

    def test_profile_length(self):
        """El perfil debe tener el número correcto de pasos."""
        assert len(self.profile.temperature_outdoor) == 24
        assert len(self.profile.occupancy) == 24
        assert len(self.profile.tariff) == 24

    def test_temperature_range(self):
        """La temperatura exterior debe estar en rango razonable."""
        temps = self.profile.temperature_outdoor
        assert np.all(temps > -10), "Temperatura demasiado baja"
        assert np.all(temps < 55), "Temperatura demasiado alta"

    def test_occupancy_non_negative(self):
        """La ocupación nunca debe ser negativa."""
        assert np.all(self.profile.occupancy >= 0)

    def test_tariff_positive(self):
        """La tarifa debe ser siempre positiva."""
        assert np.all(self.profile.tariff > 0)

    def test_solar_radiation_non_negative(self):
        """La radiación solar nunca debe ser negativa."""
        assert np.all(self.profile.solar_radiation >= 0)

    def test_normalized_tariff_range(self):
        """La tarifa normalizada debe estar en [0, 1]."""
        assert np.all(self.profile.tariff_normalized >= 0)
        assert np.all(self.profile.tariff_normalized <= 1.01)

    def test_reproducibility(self):
        """Dos perfiles con la misma semilla deben ser idénticos."""
        sim_config = SimulationConfig(horizon_hours=24, random_seed=42)
        env_config = EnvironmentConfig()
        profile2 = EnvironmentProfile(sim_config, env_config)
        np.testing.assert_array_equal(
            self.profile.temperature_outdoor, profile2.temperature_outdoor
        )

    def test_get_state_at(self):
        """get_state_at debe retornar diccionario con todas las claves necesarias."""
        state = self.profile.get_state_at(0)
        required_keys = ['time_hours', 'hour_of_day', 'temperature_outdoor',
                         'solar_radiation', 'occupancy', 'tariff',
                         'tariff_normalized', 'base_consumption']
        for key in required_keys:
            assert key in state, f"Falta clave: {key}"


class TestHouseModel:
    """Pruebas para el modelo térmico de la vivienda."""

    def setup_method(self):
        config = AppConfig()
        self.house = HouseModel(config.house, dt=1.0)

    def test_initial_temperature(self):
        """La temperatura inicial debe ser la configurada."""
        assert self.house.temperature_indoor == 26.0

    def test_step_returns_state(self):
        """Un paso debe retornar un estado válido."""
        state = self.house.step(35.0, 2, 500, 50)
        assert 'temperature_indoor' in state
        assert 'hvac_consumption_kw' in state

    def test_hvac_cools(self):
        """El HVAC a potencia alta debe enfriar la casa."""
        self.house.reset(initial_temp=30.0)
        for _ in range(10):
            self.house.step(temp_outdoor=25.0, occupancy=0,
                           solar_radiation=0, hvac_level=100)
        assert self.house.temperature_indoor < 30.0, "HVAC no enfrió"

    def test_no_hvac_heats_up(self):
        """Sin HVAC en verano, la casa debe calentarse."""
        self.house.reset(initial_temp=22.0)
        for _ in range(10):
            self.house.step(temp_outdoor=38.0, occupancy=2,
                           solar_radiation=800, hvac_level=0)
        assert self.house.temperature_indoor > 22.0, "Casa no se calentó"

    def test_temperature_bounded(self):
        """La temperatura no debe salir de rangos físicos."""
        self.house.reset(initial_temp=25.0)
        for _ in range(100):
            self.house.step(50.0, 5, 1000, 0)
        assert self.house.temperature_indoor <= 50.0

    def test_reset(self):
        """Reset debe restaurar el estado inicial."""
        self.house.step(35.0, 2, 500, 50)
        self.house.reset()
        assert self.house.temperature_indoor == 26.0
        assert self.house.hvac_power_level == 0.0


class TestScenarioGenerator:
    """Pruebas para el generador de escenarios."""

    def test_all_scenarios_valid(self):
        """Todos los escenarios predefinidos deben ser generables."""
        for scenario in AVAILABLE_SCENARIOS:
            sim_cfg, env_cfg = get_scenario_configs(scenario)
            assert sim_cfg is not None
            assert env_cfg is not None
            assert sim_cfg.scenario_type == scenario

    def test_summer_warmer_than_winter(self):
        """Verano debe tener temperatura media mayor que invierno."""
        _, summer = get_scenario_configs('verano')
        _, winter = get_scenario_configs('invierno')
        assert summer.temp_mean > winter.temp_mean


class TestSimulator:
    """Pruebas para el motor de simulación."""

    def test_simulation_runs(self):
        """La simulación debe ejecutarse sin errores."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        config.simulation.random_seed = 42
        
        from app.fuzzy.controller import FuzzyController
        controller = FuzzyController(config.fuzzy)
        simulator = Simulator(config)
        result = simulator.run(controller.get_controller_function())
        
        assert isinstance(result, SimulationResult)
        assert len(result.data) == 24  # 24 horas con paso de 1h
        assert 'temperature_indoor' in result.data.columns
        assert 'hvac_level' in result.data.columns

    def test_metrics_calculation(self):
        """Las métricas deben calcularse correctamente."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        
        from app.fuzzy.controller import FuzzyController
        controller = FuzzyController(config.fuzzy)
        simulator = Simulator(config)
        result = simulator.run(controller.get_controller_function())
        
        metrics = calculate_metrics(result.data, config.simulation, config.metrics)
        assert metrics.total_energy_kwh > 0
        assert metrics.total_cost > 0
        assert 0 <= metrics.comfort_percentage <= 100

    def test_simulator_uses_runtime_target_temperature(self):
        """La temperatura objetivo de la GUI/simulación debe propagarse al dispositivo."""
        config = AppConfig()
        config.simulation.horizon_hours = 24
        config.simulation.target_temperature = 30.0
        config.simulation.comfort_range = 1.5

        from app.fuzzy.controller import FuzzyController
        controller = FuzzyController(config.fuzzy, device_key='hvac')
        result = Simulator(config).run(controller.get_controller_function())

        assert float(result.data['target_temperature'].iloc[0]) == pytest.approx(30.0)
        assert result.data['temp_error'].iloc[0] == pytest.approx(
            result.data['temperature_indoor'].iloc[0] - 30.0,
            abs=1e-3,
        )

    def test_higher_hvac_target_reduces_cooling_demand(self):
        """Si la meta sube, el HVAC debe enfriar menos en el mismo escenario."""
        from app.fuzzy.controller import FuzzyController

        low_target_cfg = AppConfig()
        low_target_cfg.simulation.horizon_hours = 24
        low_target_cfg.simulation.scenario_type = 'verano'
        low_target_cfg.simulation.target_temperature = 22.0
        low_target_controller = FuzzyController(low_target_cfg.fuzzy, device_key='hvac')
        low_target_result = Simulator(low_target_cfg).run(low_target_controller.get_controller_function())

        high_target_cfg = AppConfig()
        high_target_cfg.simulation.horizon_hours = 24
        high_target_cfg.simulation.scenario_type = 'verano'
        high_target_cfg.simulation.target_temperature = 30.0
        high_target_controller = FuzzyController(high_target_cfg.fuzzy, device_key='hvac')
        high_target_result = Simulator(high_target_cfg).run(high_target_controller.get_controller_function())

        assert high_target_result.data['control_level'].mean() < low_target_result.data['control_level'].mean()

    def test_summer_and_winter_change_environment_profile(self):
        """El escenario seleccionado debe modificar realmente la temperatura exterior."""
        from app.fuzzy.controller import FuzzyController

        summer_cfg = AppConfig()
        summer_cfg.simulation.horizon_hours = 24
        summer_cfg.simulation.scenario_type = 'verano'
        summer_controller = FuzzyController(summer_cfg.fuzzy, device_key='hvac')
        summer_result = Simulator(summer_cfg).run(summer_controller.get_controller_function())

        winter_cfg = AppConfig()
        winter_cfg.simulation.horizon_hours = 24
        winter_cfg.simulation.scenario_type = 'invierno'
        winter_controller = FuzzyController(winter_cfg.fuzzy, device_key='hvac')
        winter_result = Simulator(winter_cfg).run(winter_controller.get_controller_function())

        assert summer_result.data['temperature_outdoor'].mean() > winter_result.data['temperature_outdoor'].mean()

    def test_winter_hvac_tracks_target_without_staying_cold(self):
        """En invierno el HVAC debe poder calentar y acercarse a la meta termica."""
        from app.fuzzy.controller import FuzzyController

        cfg = AppConfig()
        cfg.simulation.horizon_hours = 72
        cfg.simulation.scenario_type = 'invierno'
        cfg.simulation.target_temperature = 22.0
        cfg.simulation.comfort_range = 0.5

        controller = FuzzyController(cfg.fuzzy, device_key='hvac')
        result = Simulator(cfg).run(controller.get_controller_function())

        assert result.data['temperature_indoor'].mean() > 21.0
        assert 'heating' in result.data['control_mode'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
