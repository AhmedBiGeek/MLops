import os
import numpy as np
import pandas as pd
import sklearn
import mlflow
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

print("Environment is set up correctly!")
print(f"numpy {np.__version__}")
print(f"pandas {pd.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"mlflow {mlflow.__version__}")
