from decode_helpers.helpers import pca_decoder_rfe
from scipy.io import loadmat
import numpy as np
import sys
import os
from config import config
import pandas as pd


cond = sys.argv[1]
chunk = sys.argv[2]
reps = sys.argv[3]
rfun = sys.argv[4]
ndims = sys.argv[5]

if len(sys.argv) < 7:
    debug = False
else:
    debug = eval(sys.argv[6])

result_name = 'pca_decode_elimination'


if debug:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun + '_debug', 'ndims_'+ ndims)

else:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun , 'ndims_'+ ndims)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


factors = 100

if factors == 100:
    pieman_name = 'pieman_ica100.mat'
else:
    pieman_name = 'pieman_data.mat'

pieman_data = loadmat(os.path.join(config['datadir'], pieman_name))
pieman_conds = ['intact', 'paragraph', 'word', 'rest']



if debug:
    data = []
    conds = []
    for c in pieman_conds:
        next_data = list(map(lambda i: pieman_data[c][:, i][0][:30, :10], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
        data.extend(next_data)
        conds.extend([c]*len(next_data))
    del pieman_data

else:

    data = []
    conds = []
    for c in pieman_conds:
        if c == 'paragraph':
            if factors == 700:
                next_data = list(map(lambda i: pieman_data[c][:, i][0], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
            else:
                next_data = list(map(lambda i: pieman_data[c][:, i][0], np.where(np.arange(pieman_data[c].shape[1]) != 0)[0]))
        else:
            next_data = list(map(lambda i: pieman_data[c][:, i][0], np.arange(pieman_data[c].shape[1])))
        data.extend(next_data)
        conds.extend([c]*len(next_data))
    del pieman_data


data = np.array(data)
conds = np.array(conds)

append_iter = pd.DataFrame()

try_iter_results = pca_decoder_rfe(data[conds == cond], nfolds=2, dims=int(ndims))

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.scatterplot(x="dims", y="accuracy",
             data=try_iter_results, ax=ax, palette="cubehelix")
sns.despine(ax=ax, left=True)
plt.show()
print(try_iter_results)
try_iter_results['iteration'] = int(reps)


save_file = os.path.join(results_dir, cond)


if not os.path.isfile(save_file + '.csv'):
      try_iter_results.to_csv(save_file + '.csv')
else:
    append_iter = pd.read_csv(save_file + '.csv', index_col=0)
    append_iter = append_iter.append(try_iter_results)
    append_iter.to_csv(save_file + '.csv')


