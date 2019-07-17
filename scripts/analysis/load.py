import pandas as pd
from pathlib import Path
directory_of_file = Path(__file__).parent

file = directory_of_file / "data" / "16_07_2019" / "flow" / "snapshot.csv.8"
if not file.exists():
    raise Exception(f"Data file does not exist: {file}")

df = pd.read_csv(file)
df.index.name = "Bubble"