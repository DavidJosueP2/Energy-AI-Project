import numpy as np
from typing import Dict, List, Tuple, Optional


def triangular_mf(x: np.ndarray, params: List[float]) -> np.ndarray:
    """
    Función de pertenencia triangular.
    
    Definida por tres puntos [a, b, c]:
    - μ(x) = 0          si x <= a o x >= c
    - μ(x) = (x-a)/(b-a) si a < x <= b  (rampa ascendente)
    - μ(x) = (c-x)/(c-b) si b < x < c   (rampa descendente)
    - μ(x) = 1          si x == b        (pico)
    
    Args:
        x: Array de valores del universo de discurso.
        params: [a, b, c] donde a <= b <= c.
        
    Returns:
        Array de grados de pertenencia [0, 1].
    """
    a, b, c = params[0], params[1], params[2]
    
    result = np.zeros_like(x, dtype=float)
    
    if b > a:
        mask_up = (x > a) & (x <= b)
        result[mask_up] = (x[mask_up] - a) / (b - a)
    
    if c > b:
        mask_down = (x > b) & (x < c)
        result[mask_down] = (c - x[mask_down]) / (c - b)
    
    result[x == b] = 1.0
    
    if a == b:
        result[x <= a] = 1.0
    if b == c:
        result[x >= c] = 1.0
    
    return np.clip(result, 0.0, 1.0)


def trapezoidal_mf(x: np.ndarray, params: List[float]) -> np.ndarray:
    """
    Función de pertenencia trapezoidal.
    
    Definida por cuatro puntos [a, b, c, d]:
    - μ(x) = 0            si x <= a o x >= d
    - μ(x) = (x-a)/(b-a)  si a < x < b   (rampa ascendente)
    - μ(x) = 1            si b <= x <= c  (meseta)
    - μ(x) = (d-x)/(d-c)  si c < x < d   (rampa descendente)
    
    Args:
        x: Array de valores del universo de discurso.
        params: [a, b, c, d] donde a <= b <= c <= d.
        
    Returns:
        Array de grados de pertenencia [0, 1].
    """
    a, b, c, d = params[0], params[1], params[2], params[3]
    
    result = np.zeros_like(x, dtype=float)
    
    if b > a:
        mask = (x > a) & (x < b)
        result[mask] = (x[mask] - a) / (b - a)
    
    result[(x >= b) & (x <= c)] = 1.0
    
    if d > c:
        mask = (x > c) & (x < d)
        result[mask] = (d - x[mask]) / (d - c)
    
    if a == b:
        result[x <= a] = 1.0
    if c == d:
        result[x >= d] = 1.0
    
    return np.clip(result, 0.0, 1.0)


class FuzzySet:
    """
    Representa un conjunto difuso con una etiqueta lingüística
    y una función de pertenencia parametrizada.
    """
    
    def __init__(self, name: str, mf_type: str, params: List[float]):
        """
        Args:
            name: Etiqueta lingüística (ej: 'alta', 'baja').
            mf_type: Tipo de función ('triangular' o 'trapezoidal').
            params: Parámetros de la función de pertenencia.
        """
        self.name = name
        self.mf_type = mf_type
        self.params = list(params)
        
        if mf_type == 'triangular':
            self._mf_func = triangular_mf
        elif mf_type == 'trapezoidal':
            self._mf_func = trapezoidal_mf
        else:
            raise ValueError(f"Tipo de MF no soportado: {mf_type}")
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evalúa el grado de pertenencia para un array de valores."""
        return self._mf_func(x, self.params)
    
    def fuzzify(self, value: float, universe: np.ndarray) -> float:
        """
        Calcula el grado de pertenencia de un valor escalar.
        Usa interpolación sobre el universo de discurso.
        """
        membership = self._mf_func(np.array([value]), self.params)
        return float(membership[0])
    
    def get_center(self) -> float:
        """Retorna el centro de la función de pertenencia (punto de máximo)."""
        if self.mf_type == 'triangular':
            return self.params[1]
        elif self.mf_type == 'trapezoidal':
            return (self.params[1] + self.params[2]) / 2.0
        return self.params[len(self.params) // 2]
    
    def __repr__(self) -> str:
        return f"FuzzySet('{self.name}', {self.mf_type}, {self.params})"


class FuzzyVariable:
    """
    Representa una variable lingüística con su universo de discurso
    y sus conjuntos difusos asociados.
    """
    
    def __init__(self, name: str, 
                 universe_range: Tuple[float, float],
                 resolution: int = 200):
        """
        Args:
            name: Nombre de la variable (ej: 'temperatura').
            universe_range: (min, max) del universo de discurso.
            resolution: Número de puntos para discretizar el universo.
        """
        self.name = name
        self.universe_range = universe_range
        self.resolution = resolution
        self.universe = np.linspace(universe_range[0], universe_range[1], resolution)
        self.sets: Dict[str, FuzzySet] = {}
    
    def add_set(self, name: str, mf_type: str, params: List[float]) -> 'FuzzyVariable':
        """Añade un conjunto difuso a la variable. Retorna self para encadenar."""
        self.sets[name] = FuzzySet(name, mf_type, params)
        return self
    
    def add_triangular_sets(self, sets_params: Dict[str, List[float]]) -> 'FuzzyVariable':
        """Añade múltiples conjuntos triangulares de una vez."""
        for name, params in sets_params.items():
            self.add_set(name, 'triangular', params)
        return self
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzifica un valor escalar: calcula grados de pertenencia
        a todos los conjuntos difusos de la variable.
        
        Args:
            value: Valor crisp de la variable.
            
        Returns:
            Diccionario {nombre_conjunto: grado_pertenencia}.
        """
        result = {}
        for set_name, fuzzy_set in self.sets.items():
            result[set_name] = fuzzy_set.fuzzify(value, self.universe)
        return result
    
    def get_set(self, name: str) -> FuzzySet:
        """Obtiene un conjunto difuso por nombre."""
        if name not in self.sets:
            raise KeyError(f"Conjunto '{name}' no encontrado en variable '{self.name}'. "
                         f"Disponibles: {list(self.sets.keys())}")
        return self.sets[name]
    
    def get_membership_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Retorna las curvas de pertenencia para todas las funciones."""
        curves = {}
        for name, fs in self.sets.items():
            curves[name] = (self.universe.copy(), fs.evaluate(self.universe))
        return curves
    
    def validate(self) -> bool:
        """Valida que los conjuntos cubran razonablemente el universo."""
        if not self.sets:
            return False
        total_membership = np.zeros_like(self.universe)
        for fs in self.sets.values():
            total_membership += fs.evaluate(self.universe)
        coverage = np.mean(total_membership > 0.01)
        return coverage > 0.8
    
    def __repr__(self) -> str:
        sets_str = ', '.join(self.sets.keys())
        return f"FuzzyVariable('{self.name}', [{self.universe_range[0]}, {self.universe_range[1]}], sets=[{sets_str}])"
