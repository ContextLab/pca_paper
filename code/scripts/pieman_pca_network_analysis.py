from decode_helpers.helpers import pca_decoder
from scipy.io import loadmat
import numpy as np
import sys
import os
import supereeg as se
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

result_name = 'pca_decode_network'

if debug:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun + '_debug', 'ndims_'+ ndims)

else:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun , 'ndims_'+ ndims)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)



n_nets = 7

print(str(n_nets))

factors = 700

if factors == 100:
    pieman_name = 'pieman_ica100.mat'
else:
    pieman_name = 'pieman_data.mat'

pieman_data = loadmat(os.path.join(config['datadir'], pieman_name))
pieman_conds = ['intact', 'paragraph', 'word', 'rest']

network_file = os.path.join(config['datadir'],'yeo_networks', 'networks.npz')

network_npz = np.load(network_file)
network_locs = network_npz['locs']
network_data = network_npz['data']

network_list = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                'Limbic ', 'Frontoparietal', 'Default']

for n in np.arange(n_nets):
    mask_n = network_data == n + 1
    mask_n_flat = mask_n.ravel()
    data = []
    conds = []


    for c in pieman_conds:
        if c == 'paragraph':
            if factors == 700:
                next_data = list(map(lambda i: pieman_data[c][:, i][0][:, mask_n_flat], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
            else:
                next_data = list(map(lambda i: pieman_data[c][:, i][0][:, mask_n_flat], np.where(np.arange(pieman_data[c].shape[1]) != 0)[0]))
        else:
            next_data = list(map(lambda i: pieman_data[c][:, i][0][:, mask_n_flat], np.arange(pieman_data[c].shape[1])))
        data.extend(next_data)
        conds.extend([c]*len(next_data))


    data = np.array(data)
    conds = np.array(conds)

    append_iter = pd.DataFrame()

    iter_results = pca_decoder(data[conds == cond], nfolds=2, dims=int(ndims))

    print(network_list[n])
    print(iter_results)
    iter_results['iteration'] = int(reps)


    save_file = os.path.join(results_dir, cond + '_' + network_list[n])


    if not os.path.isfile(save_file + '.csv'):
          iter_results.to_csv(save_file + '.csv')
    else:
        append_iter = pd.read_csv(save_file + '.csv', index_col=0)
        append_iter = append_iter.append(iter_results)
        append_iter.to_csv(save_file + '.csv')


