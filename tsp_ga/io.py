from pathlib import Path
from .tsp import TSPInstance


def _parse_header_value(line: str) -> str:
    """Extract value from TSPLIB header lines with or without colon separator"""
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.split()[-1]


def _parse_explicit_matrix(
    values: list[float],
    dimension: int,
    edge_weight_format: str,
    *,
    source: Path,
) -> tuple[tuple[float, ...], ...]:
    """Decode EDGE_WEIGHT_SECTION values into a square distance matrix"""
    fmt = edge_weight_format.upper().strip()
    n = int(dimension)
    matrix = [[0.0] * n for _ in range(n)]

    def _expect_count(expected: int) -> None:
        """Fail fast when value count does not match declared TSPLIB format"""
        if len(values) != expected:
            raise ValueError(
                f"{source}: EDGE_WEIGHT_SECTION has {len(values)} numbers, "
                f"but format {fmt} requires {expected} for DIMENSION={n}."
            )

    idx = 0

    if fmt == "FULL_MATRIX":
        _expect_count(n * n)
        for i in range(n):
            for j in range(n):
                matrix[i][j] = float(values[idx])
                idx += 1
        return tuple(tuple(row) for row in matrix)

    if fmt == "UPPER_ROW":
        _expect_count(n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                value = float(values[idx])
                idx += 1
                matrix[i][j] = value
                matrix[j][i] = value
        return tuple(tuple(row) for row in matrix)

    if fmt == "UPPER_DIAG_ROW":
        _expect_count(n * (n + 1) // 2)
        for i in range(n):
            for j in range(i, n):
                value = float(values[idx])
                idx += 1
                matrix[i][j] = value
                matrix[j][i] = value
        return tuple(tuple(row) for row in matrix)

    if fmt == "LOWER_DIAG_ROW":
        _expect_count(n * (n + 1) // 2)
        for i in range(n):
            for j in range(i + 1):
                value = float(values[idx])
                idx += 1
                matrix[i][j] = value
                matrix[j][i] = value
        return tuple(tuple(row) for row in matrix)

    raise ValueError(
        f"{source}: unsupported EDGE_WEIGHT_FORMAT={edge_weight_format!r} for explicit matrix."
    )


def load_tsplib(path: str | Path) -> TSPInstance:
    """Load TSPLIB data from coordinate or explicit matrix sections"""

    file_path = Path(path)
    name = file_path.stem
    dimension: int | None = None
    edge_weight_type = "EUC_2D"
    edge_weight_format: str | None = None

    # Track which data section parser is currently in.
    in_coord_section = False
    in_edge_weight_section = False

    # Collected coordinates in file order.
    coordinates: list[tuple[float, float]] = []
    edge_weight_values: list[float] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("EOF"):
                break

            if upper.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                in_edge_weight_section = False
                continue

            if upper.startswith("EDGE_WEIGHT_SECTION"):
                in_coord_section = False
                in_edge_weight_section = True
                continue

            if upper.endswith("_SECTION"):
                in_coord_section = False
                in_edge_weight_section = False
                continue

            if upper.startswith("NAME"):
                name = _parse_header_value(line)
                continue

            if upper.startswith("DIMENSION"):
                dimension = int(_parse_header_value(line))
                continue

            if upper.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = _parse_header_value(line).upper()
                continue

            if upper.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = _parse_header_value(line).upper()
                continue

            if in_edge_weight_section:
                edge_weight_values.extend(float(token) for token in line.split())
                continue

            if not in_coord_section:
                continue

            parts = line.split()
            if len(parts) >= 3:
                x_token = parts[-2]
                y_token = parts[-1]
            elif len(parts) == 2:
                x_token, y_token = parts
            else:
                continue

            coordinates.append((float(x_token), float(y_token)))

    if edge_weight_type == "EXPLICIT":
        if dimension is None:
            raise ValueError(f"{file_path}: EXPLICIT instance is missing DIMENSION.")
        if edge_weight_format is None:
            raise ValueError(f"{file_path}: EXPLICIT instance is missing EDGE_WEIGHT_FORMAT.")
        explicit_weights = _parse_explicit_matrix(
            values=edge_weight_values,
            dimension=dimension,
            edge_weight_format=edge_weight_format,
            source=file_path,
        )
        return TSPInstance(
            name=name,
            coordinates=tuple(coordinates),
            edge_weight_type=edge_weight_type,
            edge_weight_format=edge_weight_format,
            explicit_weights=explicit_weights,
            dimension=dimension,
        )

    if not coordinates:
        raise ValueError(
            f"{file_path}: expected NODE_COORD_SECTION for EDGE_WEIGHT_TYPE={edge_weight_type}."
        )
    if dimension is not None and len(coordinates) != dimension:
        raise ValueError(
            f"{file_path}: parsed {len(coordinates)} coordinates but DIMENSION={dimension}."
        )

    # Return normalized immutable instance object
    return TSPInstance(
        name=name,
        coordinates=tuple(coordinates),
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        dimension=dimension,
    )
