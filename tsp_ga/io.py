from pathlib import Path
from .tsp import TSPInstance

def load_tsplib(path: str | Path) -> TSPInstance:
    """
    Load a TSPLIB-like file with a NODE_COORD_SECTION.

    Supported coordinate forms inside the section:
    - "<index> <x> <y>"
    - "<x> <y>"
    """

    file_path = Path(path)
    name = file_path.stem
    dimension: int | None = None
    edge_weight_type = "EUC_2D"

    # Track whether parser is currently inside NODE_COORD_SECTION
    in_coord_section = False

    # Collected coordinates in file order
    coordinates: list[tuple[float, float]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("NAME"):
                if ":" in line:
                    name = line.split(":", 1)[1].strip()
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                continue

            if upper.startswith("DIMENSION"):
                value = line.split(":", 1)[-1].strip() if ":" in line else line.split()[-1]
                dimension = int(value)
                continue

            if upper.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue

            if upper.startswith("EDGE_WEIGHT_TYPE"):
                value = line.split(":", 1)[-1].strip() if ":" in line else line.split()[-1]
                edge_weight_type = value.upper()
                continue

            if upper.startswith("EOF"):
                break

            if not in_coord_section:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            if len(parts) >= 3:
                x_token = parts[-2]
                y_token = parts[-1]
            else:
                x_token, y_token = parts

            coordinates.append((float(x_token), float(y_token)))

    # Return normalized immutable instance object
    return TSPInstance(
        name=name,
        coordinates=tuple(coordinates),
        edge_weight_type=edge_weight_type,
    )
