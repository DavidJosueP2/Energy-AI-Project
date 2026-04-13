"""
Base de reglas difusas para cada dispositivo.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FuzzyRule:
    """Representa una regla difusa tipo Mamdani."""

    antecedents: List[Tuple[str, str]]
    consequent: Tuple[str, str]
    weight: float = 1.0
    description: str = ""

    def __repr__(self) -> str:
        ant_str = " Y ".join(f"{var} ES {label}" for var, label in self.antecedents)
        con_str = f"{self.consequent[0]} ES {self.consequent[1]}"
        return f"SI {ant_str} ENTONCES {con_str} (w={self.weight:.2f})"


class RuleSet:
    """Contenedor interpretable de reglas difusas."""

    def __init__(self):
        self.rules: List[FuzzyRule] = []

    def add_rule(
        self,
        antecedents: List[Tuple[str, str]],
        consequent: Tuple[str, str],
        weight: float = 1.0,
        description: str = "",
    ) -> "RuleSet":
        self.rules.append(FuzzyRule(antecedents, consequent, weight, description))
        return self

    def get_rules(self) -> List[FuzzyRule]:
        return self.rules

    @property
    def num_rules(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"RuleSet({self.num_rules} reglas)"

def create_default_rule_base(device_key: str = "hvac", output_name: str = "control_output") -> RuleSet:
    """Retorna la base de reglas por defecto del dispositivo solicitado."""
    if device_key == "refrigerador":
        return create_refrigerator_rule_base(output_name)
    return create_hvac_rule_base(output_name)


def create_hvac_rule_base(output_name: str = "control_output") -> RuleSet:
    """Base de reglas HVAC centrada en confort termico interpretable."""
    rb = RuleSet()

    # Zona confortable
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "vacia")],
        (output_name, "muy_baja"),
        1.0,
        "Zona confortable y vivienda vacia.",
    )
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "baja")],
        (output_name, "baja"),
        0.95,
        "Zona confortable con poca ocupacion.",
    )
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "media")],
        (output_name, "baja"),
        0.95,
        "Zona confortable con ocupacion media: se mantiene acondicionamiento suave.",
    )
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "alta"), ("humidity", "alta")],
        (output_name, "media"),
        0.95,
        "Zona confortable pero humedad alta y ocupacion elevada.",
    )
    rb.add_rule(
        [("temp_error", "confortable"), ("tariff", "cara")],
        (output_name, "muy_baja"),
        0.90,
        "Confort alcanzado con tarifa cara.",
    )

    # Error bajo o frio: el sistema debe calentar, pero con distinta
    # agresividad segun ocupacion y costo.
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "vacia")],
        (output_name, "baja"),
        0.90,
        "Ambiente frio con vivienda vacia: calentamiento minimo de mantenimiento.",
    )
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "baja")],
        (output_name, "media"),
        0.95,
        "Ambiente frio con poca ocupacion: calentamiento moderado.",
    )
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "media")],
        (output_name, "alta"),
        1.0,
        "Ambiente frio con ocupacion media: se requiere recuperacion termica clara.",
    )
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Ambiente frio con ocupacion alta: prioridad de confort.",
    )
    rb.add_rule(
        [("temp_error", "baja"), ("tariff", "cara"), ("occupancy", "vacia")],
        (output_name, "muy_baja"),
        0.85,
        "Ambiente frio, casa vacia y tarifa cara: se modera el calentamiento.",
    )
    rb.add_rule(
        [("temp_error", "baja"), ("humidity", "alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        0.95,
        "Ambiente frio con alta ocupacion y humedad: se preserva confort.",
    )

    # Temperatura alta
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "vacia")],
        (output_name, "baja"),
        0.85,
        "Preacondicionamiento moderado.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "baja"), ("tariff", "barata")],
        (output_name, "alta"),
        0.95,
        "Condicion calida con tarifa barata.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "media")],
        (output_name, "alta"),
        1.0,
        "Condicion calida con ocupacion media.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Condicion calida con ocupacion alta.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("humidity", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Humedad alta amplifica la demanda de enfriamiento.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("tariff", "cara"), ("occupancy", "baja")],
        (output_name, "media"),
        0.90,
        "Compromiso entre costo y confort con poca ocupacion.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("tariff", "cara"), ("occupancy", "alta")],
        (output_name, "alta"),
        0.95,
        "Se preserva confort cuando la ocupacion es alta.",
    )

    # Temperatura muy alta: el controlador debe tener autoridad real
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "vacia")],
        (output_name, "alta"),
        0.90,
        "Proteccion minima de la vivienda con calor extremo.",
    )
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "baja")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con alguna ocupacion.",
    )
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "media")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con ocupacion media.",
    )
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con ocupacion alta.",
    )
    rb.add_rule(
        [("temp_error", "muy_alta"), ("humidity", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Calor y humedad criticos.",
    )
    rb.add_rule(
        [("temp_error", "muy_alta"), ("tariff", "cara"), ("occupancy", "baja")],
        (output_name, "alta"),
        0.95,
        "Tarifa cara no debe impedir reaccion ante sobretemperatura severa.",
    )
    rb.add_rule(
        [("temp_error", "alta"), ("humidity", "media"), ("occupancy", "media")],
        (output_name, "alta"),
        0.95,
        "Respuesta sostenida en condicion calida con uso normal.",
    )

    return rb


def create_refrigerator_rule_base(output_name: str = "control_output") -> RuleSet:
    """Base de reglas para refrigerador domestico."""
    rb = RuleSet()

    # Temperatura en rango
    rb.add_rule(
        [("device_temperature", "confortable"), ("door_openings", "baja"), ("load_level", "baja")],
        (output_name, "muy_baja"),
        1.0,
        "Refrigerador estable, sin uso intenso.",
    )
    rb.add_rule(
        [("device_temperature", "confortable"), ("tariff", "cara")],
        (output_name, "baja"),
        0.95,
        "En rango y con tarifa cara.",
    )
    rb.add_rule(
        [("device_temperature", "baja")],
        (output_name, "muy_baja"),
        1.0,
        "Evitar sobre-enfriamiento.",
    )

    # Temperatura alta
    rb.add_rule(
        [("device_temperature", "alta"), ("door_openings", "baja"), ("load_level", "baja")],
        (output_name, "alta"),
        0.95,
        "Temperatura alta: se requiere enfriamiento sostenido incluso sin perturbacion fuerte.",
    )
    rb.add_rule(
        [("device_temperature", "alta"), ("door_openings", "media")],
        (output_name, "muy_alta"),
        1.0,
        "Aperturas frecuentes elevan la demanda.",
    )
    rb.add_rule(
        [("device_temperature", "alta"), ("load_level", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Carga interna alta requiere mas enfriamiento.",
    )
    rb.add_rule(
        [("device_temperature", "alta"), ("tariff", "cara"), ("door_openings", "baja")],
        (output_name, "alta"),
        0.90,
        "La tarifa cara modera menos cuando la temperatura ya esta alta.",
    )

    # Temperatura muy alta
    rb.add_rule(
        [("device_temperature", "muy_alta")],
        (output_name, "muy_alta"),
        1.0,
        "Sobretemperatura critica del refrigerador.",
    )
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("door_openings", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Apertura frecuente durante sobretemperatura.",
    )
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("load_level", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Carga alta durante sobretemperatura.",
    )
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("tariff", "cara")],
        (output_name, "alta"),
        0.95,
        "La tarifa cara no anula una accion correctiva fuerte.",
    )

    return rb
