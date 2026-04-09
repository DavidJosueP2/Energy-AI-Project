# ==============================================================================
# environment.py - Generación de perfiles ambientales
# ==============================================================================
"""
Genera perfiles temporales realistas para las variables del entorno:
- Temperatura exterior (sinusoidal diaria + ruido)
- Radiación solar (campana diurna)
- Ocupación residencial (patrón de vida cotidiana)
- Tarifa eléctrica (franjas horarias)
- Consumo base del hogar (sin HVAC)

Todos los perfiles son deterministas dado una semilla, lo que garantiza
reproducibilidad completa de los experimentos.
"""

import numpy as np
from typing import Dict
from app.config import EnvironmentConfig, SimulationConfig


class EnvironmentProfile:
    """
    Genera y almacena los perfiles temporales del ambiente exterior
    y las condiciones del hogar para toda la simulación.
    """

    def __init__(self, sim_config: SimulationConfig, env_config: EnvironmentConfig):
        self.sim_config = sim_config
        self.env_config = env_config
        self.rng = np.random.RandomState(sim_config.random_seed)
        self.num_steps = sim_config.num_steps
        self.dt = sim_config.time_step_hours

        # Arrays de tiempo (en horas desde el inicio)
        self.time_hours = np.arange(self.num_steps) * self.dt

        # Generar todos los perfiles
        self.temperature_outdoor = self._generate_outdoor_temperature()
        self.humidity = self._generate_humidity()
        self.solar_radiation = self._generate_solar_radiation()
        self.occupancy = self._generate_occupancy()
        self.tariff = self._generate_tariff()
        self.tariff_normalized = self._normalize_tariff()
        self.base_consumption = self._generate_base_consumption()

    def _generate_outdoor_temperature(self) -> np.ndarray:
        """
        Genera temperatura exterior con variación sinusoidal diaria.
        
        Modelo: T_ext(t) = T_media + A * sin(2π(t - t_pico + 6)/24) + ruido
        
        El desfase de 6 horas ajusta el seno para que el pico ocurra
        a la hora configurada (típicamente 15:00).
        """
        cfg = self.env_config
        # Hora del día para cada paso
        hour_of_day = self.time_hours % 24
        
        # Componente sinusoidal: pico a las ~15h, mínimo a las ~3h
        phase = 2 * np.pi * (hour_of_day - cfg.temp_peak_hour + 6) / 24
        temp_base = cfg.temp_mean + cfg.temp_amplitude * np.sin(phase)
        
        # Ruido gaussiano suave para realismo
        noise = self.rng.normal(0, cfg.temp_noise_std, self.num_steps)
        
        # Suavizar el ruido con media móvil para evitar saltos bruscos
        kernel_size = max(1, int(3 / self.dt))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            noise = np.convolve(noise, kernel, mode='same')
        
        return temp_base + noise

    def _generate_solar_radiation(self) -> np.ndarray:
        """
        Genera radiación solar con forma de campana durante horas diurnas.
        
        Modelo: R(t) = R_max * sin²(π * (t - t_sunrise) / duración_día)
        para t entre sunrise y sunset, 0 en otro caso.
        """
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        day_duration = cfg.sunset_hour - cfg.sunrise_hour
        
        radiation = np.zeros(self.num_steps)
        
        for i, h in enumerate(hour_of_day):
            if cfg.sunrise_hour <= h <= cfg.sunset_hour:
                # Posición normalizada en el día [0, 1]
                t_norm = (h - cfg.sunrise_hour) / day_duration
                # Campana sinusoidal al cuadrado
                radiation[i] = cfg.solar_max * np.sin(np.pi * t_norm) ** 2
            else:
                radiation[i] = 0.0
        
        # Añadir variabilidad por nubes (±15%)
        cloud_factor = 1.0 + self.rng.uniform(-0.15, 0.10, self.num_steps)
        radiation *= np.clip(cloud_factor, 0.0, None)
        
        return radiation

    def _generate_humidity(self) -> np.ndarray:
        """
        Genera perfil de humedad relativa [0, 1].
        Modelo: inverso a la temperatura (humedad mas alta de noche y madrugada).
        H(t) = H_media - A_h * sin(2pi(t - t_pico + 6)/24) + ruido
        """
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        phase = 2 * np.pi * (hour_of_day - cfg.temp_peak_hour + 6) / 24
        humidity_base = cfg.humidity_mean - cfg.humidity_amplitude * np.sin(phase)
        noise = self.rng.normal(0, cfg.humidity_noise_std, self.num_steps)
        return np.clip(humidity_base + noise, 0.05, 0.99)

    def _generate_occupancy(self) -> np.ndarray:
        """
        Genera patrón de ocupación residencial típico.
        
        Perfil basado en patrones reales de vida doméstica:
        - 00:00-06:00: todos duermen (ocupación alta estable)
        - 06:00-08:00: despertar progresivo
        - 08:00-13:00: casa semi-vacía (trabajo/escuela)
        - 13:00-15:00: retorno parcial (almuerzo)
        - 15:00-17:00: baja ocupación
        - 17:00-21:00: retorno progresivo, máxima actividad
        - 21:00-00:00: preparación para dormir
        """
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        max_occ = cfg.max_occupants
        
        occupancy = np.zeros(self.num_steps)
        
        for i, h in enumerate(hour_of_day):
            if 0 <= h < 6:
                # Noche: todos en casa durmiendo
                base = max_occ * 0.95
            elif 6 <= h < 7:
                # Despertar temprano
                base = max_occ * 0.85
            elif 7 <= h < 8:
                # Preparándose para salir
                base = max_occ * 0.65
            elif 8 <= h < 9:
                # Salida progresiva
                base = max_occ * 0.35
            elif 9 <= h < 13:
                # Mañana: casa casi vacía
                base = max_occ * 0.15
            elif 13 <= h < 14:
                # Almuerzo: retorno parcial
                base = max_occ * 0.45
            elif 14 <= h < 17:
                # Tarde: baja ocupación
                base = max_occ * 0.20
            elif 17 <= h < 18:
                # Inicio retorno
                base = max_occ * 0.50
            elif 18 <= h < 20:
                # Cena y actividad familiar
                base = max_occ * 0.85
            elif 20 <= h < 22:
                # Noche en casa
                base = max_occ * 0.90
            else:  # 22-24
                # Preparándose para dormir
                base = max_occ * 0.95
            
            # Añadir variabilidad discreta
            noise = self.rng.choice([-1, 0, 0, 0, 1]) * 0.3
            occupancy[i] = np.clip(base + noise, 0, max_occ)
        
        return np.round(occupancy, 1)

    def _generate_tariff(self) -> np.ndarray:
        """
        Genera tarifa eléctrica por franjas horarias.
        
        Estructura tarifaria típica:
        - Valle (00:00-08:00): tarifa más baja
        - Llano (08:00-10:00, 14:00-18:00, 22:00-24:00): tarifa intermedia
        - Punta (10:00-14:00, 18:00-22:00): tarifa más alta
        """
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        tariff = np.zeros(self.num_steps)
        
        for i, h in enumerate(hour_of_day):
            if 0 <= h < 8:
                tariff[i] = cfg.tariff_off_peak
            elif 8 <= h < 10:
                tariff[i] = cfg.tariff_mid_peak
            elif 10 <= h < 14:
                tariff[i] = cfg.tariff_on_peak
            elif 14 <= h < 18:
                tariff[i] = cfg.tariff_mid_peak
            elif 18 <= h < 22:
                tariff[i] = cfg.tariff_on_peak
            else:  # 22-24
                tariff[i] = cfg.tariff_mid_peak
        
        return tariff

    def _normalize_tariff(self) -> np.ndarray:
        """Normaliza la tarifa al rango [0, 1] para uso en el controlador difuso."""
        t_min = self.env_config.tariff_off_peak
        t_max = self.env_config.tariff_on_peak
        if t_max == t_min:
            return np.zeros(self.num_steps)
        return (self.tariff - t_min) / (t_max - t_min)

    def _generate_base_consumption(self) -> np.ndarray:
        """
        Genera consumo eléctrico base del hogar (sin HVAC).
        
        Patrón con picos en mañana (electrodomésticos, cocina)
        y noche (iluminación, entretenimiento, cocina).
        """
        cfg = self.env_config
        hour_of_day = self.time_hours % 24
        c_min = cfg.base_consumption_min
        c_max = cfg.base_consumption_max
        c_range = c_max - c_min
        
        consumption = np.zeros(self.num_steps)
        
        for i, h in enumerate(hour_of_day):
            if 0 <= h < 6:
                # Madrugada: consumo mínimo (standby)
                factor = 0.05
            elif 6 <= h < 8:
                # Mañana temprano: iluminación, cocina
                factor = 0.55
            elif 8 <= h < 10:
                # Mañana: electrodomésticos
                factor = 0.70
            elif 10 <= h < 13:
                # Media mañana: bajo
                factor = 0.25
            elif 13 <= h < 15:
                # Almuerzo: cocina
                factor = 0.60
            elif 15 <= h < 18:
                # Tarde: bajo
                factor = 0.20
            elif 18 <= h < 21:
                # Noche: pico máximo (iluminación + cocina + entretenimiento)
                factor = 0.95
            elif 21 <= h < 23:
                # Noche tardía
                factor = 0.50
            else:
                factor = 0.15
            
            noise = self.rng.uniform(-0.05, 0.05)
            consumption[i] = c_min + c_range * np.clip(factor + noise, 0, 1)
        
        return consumption

    def get_state_at(self, step: int) -> Dict[str, float]:
        """Retorna el estado ambiental completo en un paso dado."""
        return {
            'time_hours': self.time_hours[step],
            'hour_of_day': self.time_hours[step] % 24,
            'temperature_outdoor': self.temperature_outdoor[step],
            'humidity': self.humidity[step],
            'solar_radiation': self.solar_radiation[step],
            'occupancy': self.occupancy[step],
            'tariff': self.tariff[step],
            'tariff_normalized': self.tariff_normalized[step],
            'base_consumption': self.base_consumption[step],
        }

    def get_scenario_name(self) -> str:
        """Retorna nombre descriptivo del escenario."""
        return (f"{self.sim_config.scenario_type}_"
                f"{self.sim_config.horizon_hours}h_"
                f"seed{self.sim_config.random_seed}")
