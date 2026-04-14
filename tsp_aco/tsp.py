from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class TSPInstance:
    """TSP instance represented as coordinates and distance metric."""

    name: str
    # Ordered list of node coordinates, indexed from 0 in this codebase
    coordinates: tuple[tuple[float, float], ...] = ()
    edge_weight_type: str = "EUC_2D"
    edge_weight_format: str | None = None
    explicit_weights: tuple[tuple[float, ...], ...] | None = None
    dimension: int | None = None

    @property
    def size(self) -> int:
        """Return city count regardless of whether data comes from coords or matrix"""
        if self.explicit_weights is not None:
            return len(self.explicit_weights)
        if self.dimension is not None:
            return int(self.dimension)
        # Return the number of cities in this TSP instance
        return len(self.coordinates)


# Convert TSPLIB GEO coordinate encoding to radians
def _geo_to_radians(value: float) -> float:
    """Convert TSPLIB GEO coordinate format into radians used by distance formulas"""
    # Extract degree component (integer part)
    degrees = int(value)

    # Extract minute-like fractional component
    minutes = value - degrees

    # Apply TSPLIB GEO conversion formula
    return math.pi * (degrees + 5.0 * minutes / 3.0) / 180.0


# Compute edge distance according to selected TSPLIB weight type.
def _edge_distance(
    a: tuple[float, float], b: tuple[float, float], edge_weight_type: str
) -> float:
    """Compute distance between two cities for supported TSPLIB metric types"""
    x1, y1 = a
    x2, y2 = b

    # Normalize metric name for case-insensitive comparison
    kind = edge_weight_type.upper()

    # Compute x delta
    dx = x1 - x2

    # Compute y delta
    dy = y1 - y2

    # Compute plain Euclidean distance
    euclidean = math.hypot(dx, dy)

    # TSPLIB EUC_2D: round to nearest integer via int(d + 0.5)
    if kind == "EUC_2D":
        # Return rounded Euclidean distance as float for matrix uniformity
        return float(int(euclidean + 0.5))
    
    # TSPLIB CEIL_2D: always round upward
    if kind == "CEIL_2D":
        # Return ceiling Euclidean distance
        return float(math.ceil(euclidean))
    
    # TSPLIB ATT: pseudo-Euclidean metric
    if kind == "ATT":
        # Compute ATT intermediate value
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)

        # Compute rounded base integer
        tij = int(rij + 0.5)

        # Apply ATT final correction rule
        return float(tij + 1 if tij < rij else tij)
    
    # TSPLIB GEO: great-circle approximation formula
    if kind == "GEO":
        latitude_i = _geo_to_radians(x1)
        longitude_i = _geo_to_radians(y1)
        latitude_j = _geo_to_radians(x2)
        longitude_j = _geo_to_radians(y2)

        q1 = math.cos(longitude_i - longitude_j)

        q2 = math.cos(latitude_i - latitude_j)

        q3 = math.cos(latitude_i + latitude_j)

        # GEO helper combination expression
        value = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)

        return float(int(6378.388 * math.acos(value) + 1.0))
    
    # Fallback: plain Euclidean distance for unknown metric types
    return euclidean


# Build full symmetric distance matrix for all city pairs
def build_distance_matrix(instance: TSPInstance) -> list[list[float]]:
    """Build a full symmetric distance matrix for the provided TSP instance"""
    if instance.explicit_weights is not None:
        return [list(row) for row in instance.explicit_weights]

    coords = instance.coordinates
    edge_weight_type = instance.edge_weight_type
    size = len(coords)

    # Allocate square matrix initialized with zeros
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            # Compute metric-aware distance for this edge
            distance = _edge_distance(coords[i], coords[j], edge_weight_type)

            # Write upper-triangle value
            matrix[i][j] = distance

            # Mirror to lower-triangle for symmetry
            matrix[j][i] = distance
    return matrix


# Compute total closed-tour length for a route
def tour_distance(route: Sequence[int], dist_matrix: Sequence[Sequence[float]]) -> float:
    """Compute closed tour length by summing distances for every consecutive edge"""
    total = 0.0
    route_len = len(route)
    for idx in range(route_len):
        a = route[idx]
        b = route[(idx + 1) % route_len]
        total += dist_matrix[a][b]
    return total
