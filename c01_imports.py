# =============================================================================
# Base imports (sets KERAS_BACKEND=torch BEFORE keras import)
# Migrated verbatim from Main_forGitHub.ipynb cells [4].
# Executed by runner.py inside the shared namespace (notebook-kernel style).
# =============================================================================

# ----------------------------------------------------------------------
# [notebook cell 4]
# ----------------------------------------------------------------------
# Import libraries - I - Data Processing
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import re
import sys
import warnings
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import unidecode
from rapidfuzz import fuzz, process as rfprocess
import subprocess


warnings.filterwarnings("ignore")
