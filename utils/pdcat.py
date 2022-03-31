from sys import argv
import pandas as pd

file = argv[1]
if file.endswith(".json"):
    df = pd.read_json(file)
elif file.endswith(".csv"):
    df = pd.read_csv(file)
else:
    print("File type not supported")
    raise ValueError(f"File type {file} not supported")
print(df)
