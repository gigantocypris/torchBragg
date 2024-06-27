"""
Check that pandas and torch can be imported in the same script

Usage:
libtbx.python $MODULES/torchBragg/experimental/pandas_torch_import.py
"""

import numpy as np
import pandas as pd
import torch

s = pd.Series(data=['One', 'Two', 'Three'], index=[1, 2, 3])
a = np.array([1, 2, 3])
b = s[a]
print(b.values)

print(torch.cuda.is_available())

