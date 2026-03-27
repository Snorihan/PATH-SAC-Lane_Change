import random


# Column indices in the Aimsun HIL trajectory .txt file (space-separated, no header)
# Written by writeTestData2File() in AAPI.cxx
_COL = {
    "vehID":       0,
    "hour":        1,
    "min":         2,
    "sec":         3,
    "ms":          4,
    "simStep":     5,
    "testVehID":   6,
    "tpStatus":    7,
    "vehType":     8,
    "linkID":      9,
    "nodeID":      10,
    "lane":        11,   # 1-based (Aimsun convention)
    "speed_mps":   12,   # m/s
    "pos_m":       13,   # m along link
    "totalDist":   14,   # m
    "sysEntryStep":  15,
    "linkEntryStep": 16,
    "driveMode":   17,
    "lCDir":       18,   # 1=left, 2=none, 3=right
    "lleadV":      19,   # m/s
    "lleadGap":    20,   # m
    "llagV":       21,
    "llagGap":     22,
    "rleadV":      23,
    "rleadGap":    24,
    "rlagV":       25,
    "rlagGap":     26,
    "frontV":      27,
    "frontGap":    28,
}


def load_rows(txt_path: str) -> list[dict]:
    """
    Parse the Aimsun HIL trajectory .txt file and return a list of row dicts.
    Each dict contains all 29 fields with their correct names and units.
    Returns an empty list if the file cannot be read.
    """
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 29:
                continue
            try:
                row = {name: _parse(parts[col], name) for name, col in _COL.items()}
                rows.append(row)
            except (ValueError, IndexError):
                continue
    return rows


def _parse(value: str, field: str):
    int_fields = {
        "vehID", "hour", "min", "sec", "ms", "simStep",
        "testVehID", "tpStatus", "vehType", "linkID", "nodeID",
        "lane", "sysEntryStep", "linkEntryStep", "driveMode", "lCDir",
    }
    return int(float(value)) if field in int_fields else float(value)


def sample_initial(rows: list[dict]):
    row = random.choice(rows)
    return row["lane"], row["pos_m"], row["speed_mps"]
