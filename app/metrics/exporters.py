import csv
from typing import List, Dict, Any

def export_to_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    if not data:
        return

    keys = set()
    for row in data:
        keys.update(row.keys())
    keys = sorted(list(keys))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            # Flatten non-scalar values minimally if needed
            safe_row = {k: v for k, v in row.items()}
            writer.writerow(safe_row)
