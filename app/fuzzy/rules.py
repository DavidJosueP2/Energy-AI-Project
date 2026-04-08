# ==============================================================================
# rules.py - Base de reglas del controlador difuso
# ==============================================================================
"""
Define la base de reglas del sistema de inferencia difusa.
Cada regla tiene la forma:
    SI (antecedente_1 ES etiqueta_1) Y (antecedente_2 ES etiqueta_2) Y ...
    ENTONCES (consecuente ES etiqueta_salida)

Las reglas reflejan el compromiso entre confort térmico y eficiencia energética:
- Priorizar confort cuando hay ocupación alta
- Moderar climatización cuando la tarifa es cara
- Reducir consumo cuando el consumo actual ya es elevado
- Evitar climatización innecesaria en zona de confort
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FuzzyRule:
    """
    Representa una regla difusa del tipo Mamdani.
    
    Attributes:
        antecedents: Lista de tuplas (variable, conjunto_difuso).
        consequent: Tupla (variable_salida, conjunto_difuso).
        weight: Peso de la regla [0, 1]. Por defecto 1.0.
        description: Descripción legible de la regla.
    """
    antecedents: List[Tuple[str, str]]  # [(nombre_variable, nombre_conjunto), ...]
    consequent: Tuple[str, str]          # (nombre_variable_salida, nombre_conjunto)
    weight: float = 1.0
    description: str = ""
    
    def __repr__(self) -> str:
        ant_str = " Y ".join(f"{var} ES {val}" for var, val in self.antecedents)
        con_str = f"{self.consequent[0]} ES {self.consequent[1]}"
        return f"SI {ant_str} ENTONCES {con_str} (w={self.weight})"


class RuleBase:
    """
    Base de reglas del controlador difuso.
    Contiene y administra todas las reglas del sistema.
    """
    
    def __init__(self):
        self.rules: List[FuzzyRule] = []
    
    def add_rule(self, antecedents: List[Tuple[str, str]], 
                 consequent: Tuple[str, str],
                 weight: float = 1.0,
                 description: str = "") -> 'RuleBase':
        """Añade una regla a la base. Retorna self para encadenar."""
        rule = FuzzyRule(antecedents, consequent, weight, description)
        self.rules.append(rule)
        return self
    
    def get_rules(self) -> List[FuzzyRule]:
        """Retorna todas las reglas."""
        return self.rules
    
    @property
    def num_rules(self) -> int:
        return len(self.rules)
    
    def __repr__(self) -> str:
        return f"RuleBase({self.num_rules} reglas)"


def create_default_rule_base() -> RuleBase:
    """
    Crea la base de reglas por defecto del controlador difuso.
    
    Las reglas cubren combinaciones relevantes de:
    - Error de temperatura (muy_frio, frio, confortable, calido, caliente, muy_caliente)
    - Ocupación (vacia, baja, media, alta)
    - Tarifa (barata, media, cara)
    - Consumo (bajo, medio, alto)
    → Salida: nivel de climatización (muy_baja, baja, media, alta, muy_alta)
    
    Filosofía de diseño:
    - Cuando la temperatura es confortable: climatización mínima
    - Cuando hace calor y hay gente: aumentar climatización
    - Cuando la tarifa es cara: moderar si no es urgente
    - Cuando el consumo ya es alto: evitar picos adicionales
    - Cuando la casa está vacía: reducir drásticamente
    
    Total: 35 reglas cubriendo las combinaciones más relevantes.
    """
    rb = RuleBase()
    
    # ===================================================================
    # GRUPO 1: Temperatura confortable → climatización mínima
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'confortable'), ('occupancy', 'vacia')],
        ('hvac_output', 'muy_baja'), 1.0,
        "Confortable y vacía → mínimo"
    )
    rb.add_rule(
        [('temp_error', 'confortable'), ('occupancy', 'baja')],
        ('hvac_output', 'muy_baja'), 1.0,
        "Confortable y poca gente → mínimo"
    )
    rb.add_rule(
        [('temp_error', 'confortable'), ('occupancy', 'media')],
        ('hvac_output', 'baja'), 0.9,
        "Confortable y media ocupación → bajo (mantener)"
    )
    rb.add_rule(
        [('temp_error', 'confortable'), ('occupancy', 'alta')],
        ('hvac_output', 'baja'), 0.9,
        "Confortable y llena → bajo (mantener)"
    )
    
    # ===================================================================
    # GRUPO 2: Temperatura fría → no climatizar (el HVAC enfría)
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'muy_frio')],
        ('hvac_output', 'muy_baja'), 1.0,
        "Muy frío → no enfriar más"
    )
    rb.add_rule(
        [('temp_error', 'frio')],
        ('hvac_output', 'muy_baja'), 1.0,
        "Frío → no enfriar"
    )
    
    # ===================================================================
    # GRUPO 3: Temperatura cálida + ocupación + tarifa
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'vacia')],
        ('hvac_output', 'muy_baja'), 1.0,
        "Cálido pero vacía → ahorro total"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'baja'), ('tariff', 'barata')],
        ('hvac_output', 'media'), 0.9,
        "Cálido, poca gente, tarifa barata → medio"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'baja'), ('tariff', 'media')],
        ('hvac_output', 'baja'), 0.9,
        "Cálido, poca gente, tarifa media → bajo"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'baja'), ('tariff', 'cara')],
        ('hvac_output', 'muy_baja'), 0.8,
        "Cálido, poca gente, tarifa cara → mínimo"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'media'), ('tariff', 'barata')],
        ('hvac_output', 'media'), 1.0,
        "Cálido, media ocupación, barata → medio"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'media'), ('tariff', 'cara')],
        ('hvac_output', 'baja'), 0.85,
        "Cálido, media ocupación, cara → bajo"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'alta'), ('tariff', 'barata')],
        ('hvac_output', 'alta'), 1.0,
        "Cálido, mucha gente, barata → alto"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('occupancy', 'alta'), ('tariff', 'cara')],
        ('hvac_output', 'media'), 0.9,
        "Cálido, mucha gente, cara → medio (compromiso)"
    )
    
    # ===================================================================
    # GRUPO 4: Temperatura caliente + ocupación + tarifa
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'vacia')],
        ('hvac_output', 'baja'), 0.7,
        "Caliente pero vacía → bajo (pre-acondicionar mínimo)"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'baja'), ('tariff', 'barata')],
        ('hvac_output', 'alta'), 1.0,
        "Caliente, poca gente, barata → alto"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'baja'), ('tariff', 'media')],
        ('hvac_output', 'media'), 0.9,
        "Caliente, poca gente, media → medio"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'baja'), ('tariff', 'cara')],
        ('hvac_output', 'baja'), 0.85,
        "Caliente, poca gente, cara → bajo (ahorro forzado)"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'media')],
        ('hvac_output', 'alta'), 1.0,
        "Caliente, media ocupación → alto"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'alta'), ('tariff', 'barata')],
        ('hvac_output', 'muy_alta'), 1.0,
        "Caliente, mucha gente, barata → máximo"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'alta'), ('tariff', 'media')],
        ('hvac_output', 'alta'), 1.0,
        "Caliente, mucha gente, media → alto"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('occupancy', 'alta'), ('tariff', 'cara')],
        ('hvac_output', 'media'), 0.9,
        "Caliente, mucha gente, cara → medio (compromiso)"
    )
    
    # ===================================================================
    # GRUPO 5: Temperatura muy caliente → priorizar confort
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'muy_caliente'), ('occupancy', 'vacia')],
        ('hvac_output', 'media'), 0.7,
        "Muy caliente, vacía → medio (proteger casa)"
    )
    rb.add_rule(
        [('temp_error', 'muy_caliente'), ('occupancy', 'baja')],
        ('hvac_output', 'alta'), 1.0,
        "Muy caliente, poca gente → alto (urgencia))"
    )
    rb.add_rule(
        [('temp_error', 'muy_caliente'), ('occupancy', 'media')],
        ('hvac_output', 'muy_alta'), 1.0,
        "Muy caliente, media ocupación → máximo"
    )
    rb.add_rule(
        [('temp_error', 'muy_caliente'), ('occupancy', 'alta')],
        ('hvac_output', 'muy_alta'), 1.0,
        "Muy caliente, mucha gente → máximo absoluto"
    )
    
    # ===================================================================
    # GRUPO 6: Reglas de consumo → moderar picos
    # ===================================================================
    rb.add_rule(
        [('temp_error', 'calido'), ('consumption', 'alto'), ('tariff', 'cara')],
        ('hvac_output', 'muy_baja'), 0.8,
        "Cálido, consumo alto, tarifa cara → mínimo (evitar pico)"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('consumption', 'alto'), ('tariff', 'cara')],
        ('hvac_output', 'baja'), 0.85,
        "Caliente, consumo alto, tarifa cara → bajo (moderar)"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('consumption', 'alto'), ('tariff', 'media')],
        ('hvac_output', 'media'), 0.9,
        "Caliente, consumo alto, tarifa media → medio"
    )
    rb.add_rule(
        [('temp_error', 'muy_caliente'), ('consumption', 'alto'), ('tariff', 'cara')],
        ('hvac_output', 'media'), 0.9,
        "Muy caliente pero consumo y tarifa altos → medio (equilibrio forzado)"
    )
    rb.add_rule(
        [('temp_error', 'calido'), ('consumption', 'bajo'), ('tariff', 'barata')],
        ('hvac_output', 'media'), 1.0,
        "Cálido, consumo bajo, barata → aprovechar para enfriar"
    )
    rb.add_rule(
        [('temp_error', 'caliente'), ('consumption', 'bajo'), ('tariff', 'barata')],
        ('hvac_output', 'muy_alta'), 1.0,
        "Caliente, consumo bajo, barata → máximo aprovechamiento"
    )
    
    return rb
