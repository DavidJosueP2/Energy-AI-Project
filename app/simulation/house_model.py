"""
Modelo térmico simplificado pero coherente de una vivienda residencial.

Ecuación de transición térmica (balance energético discretizado):
    T_int(t+1) = T_int(t) 
                 + α·dt·(T_ext(t) - T_int(t))        → conducción a través de envolvente
                 + β·dt·N_occ(t)                       → ganancia por ocupantes
                 + γ·dt·R_solar(t)                     → ganancia solar
                 - δ·dt·P_hvac(t)                      → extracción del climatizador

Justificación física:
- α (conductancia): modela las pérdidas/ganancias térmicas a través de paredes,
  ventanas y techo. Un edificio bien aislado tiene α ≈ 0.03-0.05; uno típico ≈ 0.06-0.10.
- β (ocupantes): cada persona genera ~80-120W de calor metabólico. En un volumen
  de aire de ~200m³, esto produce ~0.2-0.4°C/persona/hora.
- γ (solar): la radiación solar que penetra ventanas calienta el interior.
  Depende del área acristalada y factor solar del vidrio.
- δ (HVAC): capacidad de extracción de calor del sistema, normalizada.
  Un split de 3.5kW con COP=3.2 puede reducir ~0.15-0.20°C/hora a plena potencia.
"""

import numpy as np
from typing import Dict, Optional
from app.config import HouseConfig


class HouseModel:
    """
    Modelo térmico dinámico de la vivienda.
    Mantiene el estado interno y calcula transiciones temporales.
    """

    def __init__(self, config: HouseConfig, dt: float = 1.0):
        """
        Args:
            config: Configuración del modelo térmico.
            dt: Paso temporal en horas.
        """
        self.config = config
        self.dt = dt
        
        self.temperature_indoor: float = config.initial_temperature
        self.hvac_power_level: float = 0.0  # [0, 1] nivel relativo
        self.hvac_consumption_kw: float = 0.0
        
        self._step_count: int = 0

    def reset(self, initial_temp: Optional[float] = None):
        """Reinicia el modelo al estado inicial."""
        self.temperature_indoor = initial_temp or self.config.initial_temperature
        self.hvac_power_level = 0.0
        self.hvac_consumption_kw = 0.0
        self._step_count = 0

    def step(self, 
             temp_outdoor: float,
             occupancy: float,
             solar_radiation: float,
             hvac_level: float) -> Dict[str, float]:
        """
        Avanza un paso temporal y actualiza el estado de la vivienda.
        
        Args:
            temp_outdoor: Temperatura exterior actual (°C).
            occupancy: Número de ocupantes actuales.
            solar_radiation: Radiación solar actual (W/m²).
            hvac_level: Nivel de climatización ordenado por controlador [0, 100].
            
        Returns:
            Diccionario con el estado actualizado de la vivienda.
        """
        cfg = self.config
        dt = self.dt
        
        hvac_norm = np.clip(hvac_level / 100.0, 0.0, 1.0)
        self.hvac_power_level = hvac_norm
        
        delta_conduction = cfg.alpha * dt * (temp_outdoor - self.temperature_indoor)
        
        delta_occupants = cfg.beta * dt * occupancy
        
        delta_solar = cfg.gamma * dt * (solar_radiation / 1000.0) * 10.0
        
        delta_hvac = cfg.delta * dt * hvac_norm
        
        self.temperature_indoor += (
            delta_conduction 
            + delta_occupants 
            + delta_solar 
            - delta_hvac
        )
        
        self.temperature_indoor = np.clip(self.temperature_indoor, 10.0, 50.0)
        
        if hvac_norm > 0.01:
            self.hvac_consumption_kw = (
                cfg.hvac_max_power_kw * hvac_norm / cfg.hvac_cop 
                + cfg.hvac_standby_kw
            )
        else:
            self.hvac_consumption_kw = 0.0
        
        self._step_count += 1
        
        return self.get_state()

    def get_state(self) -> Dict[str, float]:
        """Retorna el estado actual de la vivienda."""
        return {
            'temperature_indoor': round(self.temperature_indoor, 3),
            'hvac_power_level': round(self.hvac_power_level, 4),
            'hvac_consumption_kw': round(self.hvac_consumption_kw, 4),
        }

    def get_thermal_components(self,
                                temp_outdoor: float,
                                occupancy: float,
                                solar_radiation: float,
                                hvac_level: float) -> Dict[str, float]:
        """
        Calcula los componentes térmicos sin actualizar el estado.
        Útil para análisis y debugging.
        """
        cfg = self.config
        dt = self.dt
        hvac_norm = np.clip(hvac_level / 100.0, 0.0, 1.0)
        
        return {
            'delta_conduction': cfg.alpha * dt * (temp_outdoor - self.temperature_indoor),
            'delta_occupants': cfg.beta * dt * occupancy,
            'delta_solar': cfg.gamma * dt * (solar_radiation / 1000.0) * 10.0,
            'delta_hvac': cfg.delta * dt * hvac_norm,
        }
