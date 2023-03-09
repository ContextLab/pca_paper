import numpy as np
import pandas as pd
import nibabel as nib
import datawrangler as dw
import seaborn as sns

from nilearn.maskers import NiftiMasker
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA as PCA
from tqdm import tqdm
from scipy.spatial.distance import cdist

import os
import warnings
import pickle

basedir = os.path.split(os.path.split(os.getcwd())[0])[0]
datadir = os.path.join(basedir, 'data')
figdir = os.path.join(basedir, 'paper', 'figs', 'source')

scratch_dir = os.path.join(basedir, 'data', 'scratch')
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

condition_colors = {
    'intact': '#21409A',
    'paragraph': '#00A14B',
    'word': '#FFDE17',
    'rest': '#7F3F98'
}

conditions = list(condition_colors.keys())


def nii2cmu(nifti_file, mask_file=None):
    '''
    inputs:
      nifti_file: a filename of a .nii or .nii.gz file to be converted into
                  CMU format
                  
      mask_file: a filename of a .nii or .nii.gz file to be used as a mask; all
                 zero-valued voxels in the mask will be ignored in the CMU-
                 formatted output.  If ignored or set to None, no voxels will
                 be masked out.
    
    outputs:
      Y: a number-of-timepoints by number-of-voxels numpy array containing the
         image data.  Each row of Y is an fMRI volume in the original nifti
         file.
      
      R: a number-of-voxels by 3 numpy array containing the voxel locations.
         Row indices of R match the column indices in Y.
    '''
    def fullfact(dims):
        '''
        Replicates MATLAB's fullfact function (behaves the same way)
        '''
        vals = np.asmatrix(range(1, dims[0] + 1)).T
        if len(dims) == 1:
            return vals
        else:
            aftervals = np.asmatrix(fullfact(dims[1:]))
            inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
            row = 0
            for i in range(aftervals.shape[0]):
                inds[row:(row + len(vals)), 0] = vals
                inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
                row += len(vals)
            return inds
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        img = nib.load(nifti_file)
        mask = NiftiMasker(mask_strategy='background')
        if mask_file is None:
            mask.fit(nifti_file)
        else:
            mask.fit(mask_file)
    
    hdr = img.header
    S = img.get_sform()
    vox_size = hdr.get_zooms()
    im_size = img.shape
    
    if len(img.shape) > 3:
        N = img.shape[3]
    else:
        N = 1
    
    Y = np.float32(mask.transform(nifti_file)).copy()
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = fullfact(img.shape[0:3])[vmask, ::-1]-1
    
    R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]

    # center on the MNI152 brain (hard code this in)
    #mni_center = np.array([0.55741881, -21.52140703, 9.83783098])
    #R = R - R.mean(axis=0) + mni_center
    
    return {'Y': Y, 'R': R}

def cmu2nii(Y, R, template=None):
    '''
    inputs:
      Y: a number-of-timepoints by number-of-voxels numpy array containing the
         image data.  Each row of Y is an fMRI volume in the original nifti
         file.
      
      R: a number-of-voxels by 3 numpy array containing the voxel locations.
         Row indices of R match the column indices in Y.
      
      template: a filename of a .nii or .nii.gz file to be used as an image
                template.  Header information of the outputted nifti images will
                be read from the header file.  If this argument is ignored or
                set to None, header information will be inferred based on the
                R array.
    
    outputs:
      nifti_file: a filename of a .nii or .nii.gz file to be converted into
                  CMU format
                  
      mask_file: a filename for a .nii or .nii.gz file to be used as a mask; all
                 zero-valued voxels in the mask will be ignored in the CMU-
                 formatted output
    
    outputs:
      img: a nibabel Nifti1Image object containing the fMRI data
    '''
    Y = np.array(Y, ndmin=2)
    img = nib.load(template)
    S = img.affine
    locs = np.array(np.dot(R - S[:3, 3], np.linalg.inv(S[0:3, 0:3])), dtype='int')
    
    data = np.zeros(tuple(list(img.shape)[0:3]+[Y.shape[0]]))
    
    # loop over data and locations to fill in activations
    for i in range(Y.shape[0]):
        for j in range(R.shape[0]):
            data[locs[j, 0], locs[j, 1], locs[j, 2], i] = Y[i, j]
    
    return nib.Nifti1Image(data, affine=img.affine)


def group_pca(data, n_components=None, fname=None):
    if fname is not None:
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                return pickle.load(f)
    
    pca = PCA(n_components=n_components)
    
    x = dw.stack(data)
    y = pca.fit_transform(x)

    y = dw.unstack(pd.DataFrame(index=x.index, data=y))

    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump((y, pca), f)
    
    return y, pca


def accuracy(train, test):
    train = np.mean(np.stack(train, axis=2), axis=2)
    test = np.mean(np.stack(test, axis=2), axis=2)
    dists = cdist(train, test, metric='correlation')
    
    labels = np.argmin(dists, axis=1)
    return np.mean([i == d for i, d in enumerate(labels)]) - 1 / len(labels)


def cross_validation(data, n_iter=10, fname=None, max_components=700):
    if fname is not None:
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                return pickle.load(f)

    results = pd.DataFrame(columns=['Iteration', 'Number of components', 'Relative decoding accuracy'])

    n = len(data[3]) // 2
    for i in tqdm(range(n_iter)):
        order = np.random.permutation(len(data[3]))

        for c in range(3, max_components + 1):
            x = pd.DataFrame(columns=['Iteration', 'Number of components', 'Relative decoding accuracy'])
            x.loc[0, 'Iteration'] = i
            x.loc[0, 'Number of components'] = c

            train = [data[c][o] for o in order[:n]]
            test = [data[c][o] for o in order[n:]]
            x.loc[0, 'Relative decoding accuracy'] = (accuracy(train, test) + accuracy(test, train)) / 2

            results = pd.concat([results, x], ignore_index=True)
    
    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def ridge_plot(x, column='Number of components', fname=None, xlim=[-99, 700], hue='Condition', palette=[condition_colors[c] for c in conditions]):

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(x, row=hue, hue=hue, palette=palette, height=1, aspect=6)
    g.map(sns.kdeplot, column, bw_adjust=1, clip_on=True, fill=True, alpha=1, common_norm=True, linewidth=1.5)
    g.refline(y=0, linewidth=1.5, linestyle='-', color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label.capitalize(), color=color, ha='left', va='center', transform=ax.transAxes)

    g.map(label, hue)

    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    fig = plt.gcf()
    fig.set_size_inches(4, 3)

    ax = plt.gca()
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel(column, fontsize=12)

    if fname is not None:
        g.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')

    return fig


def get_data():
    url = 'https://www.dropbox.com/s/29a48lv3j5ybcvw/pieman2_htfa.pkl?dl=1'

    fname = os.path.join(datadir, 'pieman2_htfa.pkl')
    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    with open(fname, 'rb') as f:
        data = pickle.load(open(fname, 'rb'))
    
    return data


def info_and_compressibility(d, target=0.05):
    def closest(x, target):
        dists = np.abs(x.values - target)
        dists[x.values < target] += 10 * np.max(dists)
        return int(x.index.values[np.argmin(dists)])

    df = []
    for c in conditions:
        dc = d[c].astype(float).pivot(index='Iteration', columns='Number of components', values='Relative decoding accuracy')
        i = pd.DataFrame()
        i['Number of components'] = dc.apply(lambda x: closest(x, target), axis=1, raw=False)
        i['Relative decoding accuracy'] = dc.max(axis=1)
        i['Condition'] = c
        i['Iteration'] = dc.index.values.astype(int)
        df.append(i)
    return pd.concat(df, ignore_index=True, axis=0)


def plot_info_and_compressibility_scatter(x, fname=None):
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()

    x = info_and_compressibility(x)
    sns.scatterplot(x, x='Number of components', y='Relative decoding accuracy', hue='Condition', palette=[condition_colors[c] for c in conditions], legend=False, s=10, ax=ax)
    sns.scatterplot(x.groupby('Condition').mean().loc[conditions].reset_index(), x='Number of components', y='Relative decoding accuracy', hue='Condition', palette=[condition_colors[c] for c in conditions], legend=False, s=100, ax=ax)

    ax.set_xlabel('Number of components', fontsize=12)
    ax.set_ylabel('Relative decoding accuracy', fontsize=12)
    ax.set_ylim(-0.01, 0.35)
    ax.set_xlim(3, 700)

    ax.spines[['right', 'top']].set_visible(False)

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    
    return fig