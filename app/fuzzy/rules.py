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
    # Criterio experto de gestion energetica: si el error esta cerca del setpoint y no hay ocupantes,
    # se minimiza la accion. El manual Mitsubishi solo respalda que cerca del setpoint la unidad pasa
    # de "operating to reach the set temperature" a "approaching the set temperature"; no define vacia.
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "vacia")],
        (output_name, "muy_baja"),
        1.0,
        "Zona confortable y vivienda vacia.",
    )
    # Manual Mitsubishi: a 1-2 C del setpoint la habitacion "is approaching the set temperature",
    # asi que la autoridad debe ser menor que cuando el error es >= 2 C. La ocupacion baja es criterio experto.
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "baja")],
        (output_name, "baja"),
        0.95,
        "Zona confortable con poca ocupacion.",
    )
    # Manual Mitsubishi: cerca del setpoint la unidad ya no esta en recuperacion fuerte.
    # Mantener accion suave con ocupacion media es criterio experto para evitar deriva termica.
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "media")],
        (output_name, "baja"),
        0.95,
        "Zona confortable con ocupacion media: se mantiene acondicionamiento suave.",
    )
    # ASHRAE 55: el confort es una zona temperatura-humedad; humedad alta reduce aceptabilidad.
    # Ocuppacion alta como modulador es criterio experto del escenario residencial.
    rb.add_rule(
        [("temp_error", "confortable"), ("occupancy", "alta"), ("humidity", "alta")],
        (output_name, "media"),
        0.95,
        "Zona confortable pero humedad alta y ocupacion elevada.",
    )
    # Criterio experto de gestion energetica: con confort alcanzado y tarifa cara se reduce accion.
    # La tarifa no esta en el manual Mitsubishi; pertenece al objetivo del sistema, no al equipo.
    rb.add_rule(
        [("temp_error", "confortable"), ("tariff", "cara")],
        (output_name, "muy_baja"),
        0.90,
        "Confort alcanzado con tarifa cara.",
    )

    # Error bajo o frio: el sistema debe calentar, pero con distinta
    # agresividad segun ocupacion y costo.
    # Manual Mitsubishi: existe modo HEAT y el setpoint es comun para frio/calor en 16-31 C.
    # Criterio experto: si la vivienda esta vacia, el calentamiento se deja en mantenimiento.
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "vacia")],
        (output_name, "baja"),
        0.90,
        "Ambiente frio con vivienda vacia: calentamiento minimo de mantenimiento.",
    )
    # Manual Mitsubishi: una desviacion apreciable respecto al setpoint requiere recuperacion.
    # La intensidad media para ocupacion baja es criterio experto.
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "baja")],
        (output_name, "media"),
        0.95,
        "Ambiente frio con poca ocupacion: calentamiento moderado.",
    )
    # Manual Mitsubishi: a >= 2 C del setpoint la unidad "is operating to reach the set temperature".
    # Se traduce a recuperacion clara cuando el recinto esta frio y hay uso normal.
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "media")],
        (output_name, "alta"),
        1.0,
        "Ambiente frio con ocupacion media: se requiere recuperacion termica clara.",
    )
    # Manual Mitsubishi: el equipo debe recuperar el setpoint cuando la desviacion es clara.
    # Dar prioridad maxima con ocupacion alta es criterio experto de confort.
    rb.add_rule(
        [("temp_error", "baja"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Ambiente frio con ocupacion alta: prioridad de confort.",
    )
    # Criterio experto de gestion energetica: tarifa cara y vivienda vacia permiten moderar calefaccion.
    # Esto no proviene del manual del equipo.
    rb.add_rule(
        [("temp_error", "baja"), ("tariff", "cara"), ("occupancy", "vacia")],
        (output_name, "muy_baja"),
        0.85,
        "Ambiente frio, casa vacia y tarifa cara: se modera el calentamiento.",
    )
    # ASHRAE 55: el confort no depende solo de temperatura; humedad y ocupacion pueden empeorar sensacion.
    # La autoridad maxima en esta combinacion es una decision experta.
    rb.add_rule(
        [("temp_error", "baja"), ("humidity", "alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        0.95,
        "Ambiente frio con alta ocupacion y humedad: se preserva confort.",
    )

    # Temperatura alta
    # Manual Mitsubishi: cuando hay error respecto al setpoint el equipo opera para alcanzarlo.
    # Con vivienda vacia se hace preacondicionamiento moderado por criterio experto.
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "vacia")],
        (output_name, "baja"),
        0.85,
        "Preacondicionamiento moderado.",
    )
    # Manual Mitsubishi: a >= 2 C del setpoint la unidad sigue corrigiendo.
    # Tarifa barata como permiso para enfriamiento mas alto es criterio de gestion energetica.
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "baja"), ("tariff", "barata")],
        (output_name, "alta"),
        0.95,
        "Condicion calida con tarifa barata.",
    )
    # Manual Mitsubishi: desviacion positiva clara implica accion para recuperar setpoint.
    # Ocuppacion media es modulador experto del uso del recinto.
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "media")],
        (output_name, "alta"),
        1.0,
        "Condicion calida con ocupacion media.",
    )
    # Manual Mitsubishi: recuperacion fuerte cuando la habitacion esta claramente alejada del setpoint.
    # Priorizar confort con alta ocupacion es criterio experto.
    rb.add_rule(
        [("temp_error", "alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Condicion calida con ocupacion alta.",
    )
    # ASHRAE 55: la aceptabilidad se define en una zona temperatura-humedad; humedad alta desplaza percepcion.
    # Se usa para justificar mayor enfriamiento cuando la humedad es alta.
    rb.add_rule(
        [("temp_error", "alta"), ("humidity", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Humedad alta amplifica la demanda de enfriamiento.",
    )
    # Criterio experto de compromiso costo-confort: con poca ocupacion y tarifa cara se evita el maximo.
    # La tarifa no viene del fabricante.
    rb.add_rule(
        [("temp_error", "alta"), ("tariff", "cara"), ("occupancy", "baja")],
        (output_name, "media"),
        0.90,
        "Compromiso entre costo y confort con poca ocupacion.",
    )
    # Criterio experto: aunque la tarifa sea cara, alta ocupacion mantiene demanda de confort relevante.
    rb.add_rule(
        [("temp_error", "alta"), ("tariff", "cara"), ("occupancy", "alta")],
        (output_name, "alta"),
        0.95,
        "Se preserva confort cuando la ocupacion es alta.",
    )

    # Temperatura muy alta: el controlador debe tener autoridad real
    # Manual Mitsubishi: "operating to reach the set temperature" cuando la habitacion esta >= 2 C lejos.
    # En sobretemperatura severa incluso vivienda vacia requiere accion de proteccion minima.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "vacia")],
        (output_name, "alta"),
        0.90,
        "Proteccion minima de la vivienda con calor extremo.",
    )
    # Manual Mitsubishi: gran desviacion del setpoint requiere autoridad alta de compresor.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "baja")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con alguna ocupacion.",
    )
    # Manual Mitsubishi: gran desviacion del setpoint requiere autoridad alta de compresor.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "media")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con ocupacion media.",
    )
    # Manual Mitsubishi: gran desviacion del setpoint requiere autoridad alta de compresor.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("occupancy", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Calor extremo con ocupacion alta.",
    )
    # ASHRAE 55 y confort practico: calor mas humedad empeoran claramente la sensacion termica.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("humidity", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Calor y humedad criticos.",
    )
    # Criterio experto de gestion energetica: la tarifa cara puede moderar, pero no anular, una correccion severa.
    rb.add_rule(
        [("temp_error", "muy_alta"), ("tariff", "cara"), ("occupancy", "baja")],
        (output_name, "alta"),
        0.95,
        "Tarifa cara no debe impedir reaccion ante sobretemperatura severa.",
    )
    # ASHRAE respalda que el confort es zona termo-higrometrica; mantener respuesta sostenida en uso normal
    # es una decision experta del controlador, no una cita literal del manual.
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
    # Manual Bosch: la temperatura recomendada del refrigerador es 4 C y el rango ajustable es 2-8 C.
    # Si el compartimento esta en rango y no hay uso intenso, se minimiza la accion.
    rb.add_rule(
        [("device_temperature", "confortable"), ("door_openings", "baja"), ("load_level", "baja")],
        (output_name, "muy_baja"),
        1.0,
        "Refrigerador estable, sin uso intenso.",
    )
    # Criterio experto de gestion energetica: si la temperatura ya esta en rango, una tarifa cara permite bajar accion.
    # La tarifa no viene del manual Bosch.
    rb.add_rule(
        [("device_temperature", "confortable"), ("tariff", "cara")],
        (output_name, "baja"),
        0.95,
        "En rango y con tarifa cara.",
    )
    # Manual Bosch: el compartimento puede ajustarse entre 2 C y 8 C; por debajo del objetivo conviene evitar
    # sobre-enfriamiento para no salir del rango de uso previsto.
    rb.add_rule(
        [("device_temperature", "baja")],
        (output_name, "muy_baja"),
        1.0,
        "Evitar sobre-enfriamiento.",
    )

    # Temperatura alta
    # Manual Bosch: 4 C es la referencia recomendada; si la temperatura sube por encima del rango objetivo,
    # el equipo debe enfriar de forma sostenida aun sin perturbacion fuerte.
    rb.add_rule(
        [("device_temperature", "alta"), ("door_openings", "baja"), ("load_level", "baja")],
        (output_name, "alta"),
        0.95,
        "Temperatura alta: se requiere enfriamiento sostenido incluso sin perturbacion fuerte.",
    )
    # Manual Bosch: puerta abierta por tiempo prolongado provoca aumento considerable de temperatura.
    rb.add_rule(
        [("device_temperature", "alta"), ("door_openings", "media")],
        (output_name, "muy_alta"),
        1.0,
        "Aperturas frecuentes elevan la demanda.",
    )
    # Manual Bosch: al introducir grandes cantidades de alimentos se recomienda Super cooling / Super freezing.
    # Esto respalda mayor enfriamiento cuando la carga interna es alta.
    rb.add_rule(
        [("device_temperature", "alta"), ("load_level", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Carga interna alta requiere mas enfriamiento.",
    )
    # Criterio experto de gestion energetica: la tarifa cara modera, pero no elimina, la respuesta si ya esta caliente.
    rb.add_rule(
        [("device_temperature", "alta"), ("tariff", "cara"), ("door_openings", "baja")],
        (output_name, "alta"),
        0.90,
        "La tarifa cara modera menos cuando la temperatura ya esta alta.",
    )

    # Temperatura muy alta
    # Manual Bosch: la alarma de temperatura se activa cuando el congelador esta demasiado caliente;
    # por analogia operativa, una sobretemperatura critica del compartimento requiere respuesta maxima.
    rb.add_rule(
        [("device_temperature", "muy_alta")],
        (output_name, "muy_alta"),
        1.0,
        "Sobretemperatura critica del refrigerador.",
    )
    # Manual Bosch: puerta abierta aumenta considerablemente la temperatura del compartimento.
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("door_openings", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Apertura frecuente durante sobretemperatura.",
    )
    # Manual Bosch: para grandes cantidades de alimento fresco se recomienda activar Super cooling/Freezing.
    # Eso justifica enfriamiento maximo con carga alta y sobretemperatura.
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("load_level", "alta")],
        (output_name, "muy_alta"),
        1.0,
        "Carga alta durante sobretemperatura.",
    )
    # Criterio experto de gestion energetica: la tarifa cara no debe cancelar una accion correctiva fuerte
    # cuando ya existe riesgo de salir demasiado del rango de conservacion.
    rb.add_rule(
        [("device_temperature", "muy_alta"), ("tariff", "cara")],
        (output_name, "alta"),
        0.95,
        "La tarifa cara no anula una accion correctiva fuerte.",
    )

    return rb
