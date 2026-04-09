# ==============================================================================
# devices.py - Modelos genericos de dispositivos controlados
# ==============================================================================
"""
Define un modelo abstracto de dispositivo controlado y dos implementaciones
concretas: HVAC (climatizacion) y Refrigerador.

Cada dispositivo:
- Tiene parametros termicos propios
- Usa la misma logica difusa para determinar su nivel de operacion
- Consume energia de forma diferente
- Tiene su propio rango de temperatura objetivo
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DeviceConfig:
    """Configuracion base de un dispositivo controlado."""
    name: str = "Dispositivo"
    display_name: str = "Dispositivo Generico"

    # Temperatura objetivo del dispositivo
    target_temperature: float = 22.0
    comfort_range: float = 2.0

    # Parametros del modelo termico
    alpha: float = 0.08    # Conductancia termica (algo de intercambio con exterior)
    beta: float = 0.05     # Calor por ocupante (4 ocupantes = 0.2 C/h)
    gamma: float = 0.02    # Ganancia solar
    delta: float = 3.5     # Efecto de enfriamiento del HVAC (a 100% enfría ~3.5 C/h)

    # Parametros electricos
    max_power_kw: float = 3.5

    cop: float = 3.2
    standby_kw: float = 0.15

    # Temperatura inicial
    initial_temperature: float = 26.0

    # Rango de temperatura valido
    temp_min: float = 10.0
    temp_max: float = 50.0


def get_hvac_config() -> DeviceConfig:
    """Configuracion para sistema HVAC residencial."""
    return DeviceConfig(
        name="hvac",
        display_name="HVAC (Climatizacion)",
        target_temperature=22.0,
        comfort_range=2.0,
        alpha=0.08,
        beta=0.05,
        gamma=0.02,
        delta=3.5,
        max_power_kw=3.5,
        cop=3.2,
        standby_kw=0.15,
        initial_temperature=26.0,
        temp_min=10.0,
        temp_max=50.0,
    )


def get_refrigerator_config() -> DeviceConfig:
    """
    Configuracion para refrigerador domestico.

    Modelo simplificado:
    - Temperatura objetivo: 4 grados (rango de refrigeracion)
    - Aislamiento mejor que una habitacion (alpha menor)
    - Sin ocupantes ni ganancia solar directa
    - Potencia mucho menor que HVAC
    - COP tipico de refrigerador domestico
    """
    return DeviceConfig(
        name="refrigerador",
        display_name="Refrigerador",
        target_temperature=4.0,
        comfort_range=1.5,
        alpha=0.025,      # Mejor aislamiento
        beta=0.0,          # Sin ocupantes
        gamma=0.0,         # Sin ganancia solar
        delta=0.35,        # Mayor efecto por kW (volumen pequeno)
        max_power_kw=0.25, # Potencia tipica compresor domestico
        cop=2.5,           # COP menor que HVAC
        standby_kw=0.03,   # Standby minimo
        initial_temperature=8.0,
        temp_min=-5.0,
        temp_max=30.0,
    )


AVAILABLE_DEVICES = {
    'hvac': get_hvac_config,
    'refrigerador': get_refrigerator_config,
}


class ControlledDevice:
    """
    Modelo generico de un dispositivo controlado por logica difusa.

    El dispositivo mantiene una variable de estado (temperatura)
    y la actualiza segun la ecuacion de transicion termica.
    """

    def __init__(self, config: DeviceConfig, dt: float = 1.0):
        """
        Args:
            config: Configuracion del dispositivo.
            dt: Paso temporal en horas.
        """
        self.config = config
        self.dt = dt

        # Estado actual
        self.temperature: float = config.initial_temperature
        self.power_level: float = 0.0
        self.consumption_kw: float = 0.0

    def reset(self, initial_temp: Optional[float] = None):
        """Reinicia el dispositivo al estado inicial."""
        self.temperature = initial_temp or self.config.initial_temperature
        self.power_level = 0.0
        self.consumption_kw = 0.0

    def step(self,
             temp_ambient: float,
             occupancy: float,
             solar_radiation: float,
             control_level: float) -> Dict[str, float]:
        """
        Avanza un paso temporal y actualiza el estado.

        Args:
            temp_ambient: Temperatura del ambiente externo.
            occupancy: Numero de ocupantes (0 para refrigerador).
            solar_radiation: Radiacion solar (0 para refrigerador).
            control_level: Nivel de control ordenado [0, 100].

        Returns:
            Estado actualizado del dispositivo.
        """
        cfg = self.config
        dt = self.dt

        # Normalizar control a [0, 1]
        ctrl_norm = np.clip(control_level / 100.0, 0.0, 1.0)
        self.power_level = ctrl_norm

        # Componentes de la transicion termica
        delta_conduction = cfg.alpha * dt * (temp_ambient - self.temperature)
        delta_occupants = cfg.beta * dt * occupancy
        delta_solar = cfg.gamma * dt * (solar_radiation / 1000.0) * 10.0
        delta_cooling = cfg.delta * dt * ctrl_norm

        # Actualizar temperatura
        self.temperature += (
            delta_conduction
            + delta_occupants
            + delta_solar
            - delta_cooling
        )

        # Limitar a rango fisico
        self.temperature = np.clip(self.temperature, cfg.temp_min, cfg.temp_max)

        # Calcular consumo electrico
        if ctrl_norm > 0.01:
            self.consumption_kw = (
                cfg.max_power_kw * ctrl_norm / cfg.cop
                + cfg.standby_kw
            )
        else:
            self.consumption_kw = 0.0

        return self.get_state()

    def get_state(self) -> Dict[str, float]:
        """Retorna el estado actual del dispositivo."""
        return {
            'device_temperature': round(self.temperature, 3),
            'device_power_level': round(self.power_level, 4),
            'device_consumption_kw': round(self.consumption_kw, 4),
        }

    def get_temp_error(self) -> float:
        """Calcula el error de temperatura respecto al objetivo."""
        return self.temperature - self.config.target_temperature

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def display_name(self) -> str:
        return self.config.display_name
