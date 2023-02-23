import sys
import os
#sys.path.append(sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#sys.path.append(os.path.abspath('..'))
from decode_helpers.helpers import pca_decoder
from scipy.io import loadmat
import numpy as np
import sys
import os
from config import config
import pandas as pd
import hypertools as hyp

cond = sys.argv[1]


print(cond)
print('done')