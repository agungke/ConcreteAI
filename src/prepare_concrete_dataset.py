"""Download and validate UCI Concrete Compressive Strength data (Yeh, 1998)."""
import argparse, csv, io, math
from pathlib import Path
from urllib.request import urlopen

URL = "https://archive.ics.uci.edu/static/public/165/data.csv"
ORIGINAL = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength"]
COLUMNS = ["cement_kg_m3", "slag_kg_m3", "fly_ash_kg_m3", "water_kg_m3", "superplasticizer_kg_m3", "coarse_aggregate_kg_m3", "fine_aggregate_kg_m3", "age_days", "strength_mpa"]

def prepare(raw, output):
    reader = csv.reader(io.StringIO(raw))
    header = next(reader, [])
    if [x.strip() for x in header] != ORIGINAL:
        raise ValueError(f"Unexpected source columns: {header}")
    rows = list(reader)
    if len(rows) != 1030:
        raise ValueError(f"Expected 1030 rows, received {len(rows)}")
    numeric = []
    for i, row in enumerate(rows, 2):
        if len(row) != 9: raise ValueError(f"Invalid column count at line {i}")
        values = tuple(float(x) for x in row)
        if not all(math.isfinite(x) for x in values): raise ValueError(f"Missing or non-finite value at line {i}")
        if any(x < 0 for x in values[:7]) or not 1 <= values[7] <= 365 or not values[7].is_integer() or values[8] <= 0:
            raise ValueError(f"Out-of-range value at line {i}")
        numeric.append(values)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(COLUMNS); writer.writerows(rows)
    print(f"Created {output}: {len(rows)} rows, 9 columns")
    print("Missing/non-finite: 0; exact duplicate rows:", len(numeric)-len(set(numeric)))
    print("Duplicates are retained and must be considered during group-aware splitting.")
    print("Source: https://doi.org/10.24432/C5PK67 (CC BY 4.0)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", default="data/concrete_strength.csv")
    args = parser.parse_args()
    if args.input: raw = args.input.read_text(encoding="utf-8-sig")
    else:
        with urlopen(URL, timeout=45) as response: raw = response.read().decode("utf-8-sig")
    prepare(raw, args.output)
