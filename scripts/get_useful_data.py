import pandas as pd
from pathlib import Path

use_data = pd.read_csv(Path.cwd()/'data'/'data.csv', usecols=[3, 4], converters={'text': pd.eval})

use_data.to_csv(Path.cwd()/'data'/'use_data.csv', index=False)
