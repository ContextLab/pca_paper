
import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine
from scipy.io import loadmat
import numpy as np
import sys
import os
from config import config
import pandas as pd
import supereeg as se

cond = sys.argv[1]
chunk = sys.argv[2]
reps = sys.argv[3]
cfun = sys.argv[4]
rfun = sys.argv[5]
width = int(sys.argv[6])
wp = sys.argv[7]
ndims = sys.argv[8]

if len(sys.argv) < 10:
    debug = False
else:
    debug = eval(sys.argv[9])

result_name = 'pca_decode_network_chunked'

if debug:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun + '_debug', 'ndims_'+ ndims)

else:
    results_dir = os.path.join(config['resultsdir'], result_name, rfun , 'ndims_'+ ndims)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)



laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta = {'name': '$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}

n_nets = 7

print(str(n_nets))

factors = 700

if factors == 100:
    pieman_name = 'pieman_ica100.mat'
else:
    pieman_name = 'pieman_data.mat'

pieman_data = loadmat(os.path.join(config['datadir'], pieman_name))
pieman_conds = ['intact', 'paragraph', 'word', 'rest']

network_bo_file = os.path.join(config['datadir'], 'networks.bo')
network_bo = se.load(network_bo_file)
network_list = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                'Limbic ', 'Frontoparietal', 'Default']

weights_paramter = eval(wp)

chunks = 3

for n in np.arange(n_nets):
    mask_n = network_bo.get_data() == n + 1
    mask_n_flat = mask_n.as_matrix().ravel()

    data_chunks = [0] * chunks
    conds_chunks = [0] * chunks
    for third in list(range(chunks)):

        data = []
        conds = []

        for c in pieman_conds:
            timechunk = int(np.round(pieman_data[c][0][0].shape[0]/3, 0))
            divided = third * timechunk
            if c == 'paragraph':
                if factors == 700:
                    next_data = list(
                        map(lambda i: pieman_data[c][:, i][0][divided:divided+timechunk,mask_n_flat], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
                else:
                    next_data = list(
                        map(lambda i: pieman_data[c][:, i][0][divided:divided+timechunk,mask_n_flat], np.where(np.arange(pieman_data[c].shape[1]) != 0)[0]))
            else:
                next_data = list(map(lambda i: pieman_data[c][:, i][0][divided:divided+timechunk,mask_n_flat], np.arange(pieman_data[c].shape[1])))
            print(np.shape(next_data))
            data.extend(next_data)
            conds.extend([c]*len(next_data))

        conds_chunks[third] = conds
        data_chunks[third] = data


    for chunk in range(chunks):

        data = np.array(data_chunks[chunk])
        conds = np.array(conds_chunks[chunk])

        append_iter = pd.DataFrame()

        iter_results = tc.helpers.pca_decoder(data[conds == cond], nfolds=2, dims=int(ndims),
                                            combine=mean_combine,
                                            cfun=eval(cfun),
                                            rfun=None,
                                            weights_fun=weights_paramter['weights'],
                                            weights_params=weights_paramter['params'])

        print(iter_results)
        iter_results['iteration'] = int(reps)
        iter_results['third'] = int(chunk)

        save_file = os.path.join(results_dir, cond + '_' + network_list[n])



        if not os.path.isfile(save_file + '.csv'):
              iter_results.to_csv(save_file + '.csv')
        else:
            append_iter = pd.read_csv(save_file + '.csv', index_col=0)
            append_iter = append_iter.append(iter_results)
            append_iter.to_csv(save_file + '.csv')


