import numpy as np
import matplotlib.pyplot as plt
from qSMLM.utils import computeCov

with open('cov.json', 'r') as f:
    config = json.load(f)

prefixes = [
]
