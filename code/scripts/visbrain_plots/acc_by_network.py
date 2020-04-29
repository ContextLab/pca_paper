import supereeg as se
import os
import numpy as np
from visbrain.objects import BrainObj, SceneObj, SourceObj
from visbrain.gui import Brain

#cmap = "yeo_colors_7_l"
#cmap = "yeo_colors_7"


cmap = "black_7"
template_brain = 'B3'

CBAR_STATE = dict(cbtxtsz=12, clim=[0, 7], txtsz=10., width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=1)

ddir = '../../../data/'

results_dir = os.path.join(ddir, 'results')

nii_bo_dir = '../../../data/niis/networks'

fig_dir = '../../../paper/figs/source/networks'


acc_fname = os.path.join(results_dir, 'accuracy_by_network.npz')
acc_file = np.load(acc_fname)


hills = acc_file['hills'][:, 0, 0]
maxs = acc_file['maxs'][:, 0, 0]

def norm_vals(vals):
    return (vals-vals.min())/(vals.max() - vals.min())




b_yeo = se.load(os.path.join(nii_bo_dir, 'yeo_bo_6mm.bo'))

data_yeo = b_yeo.get_data().values.ravel()
xyz_yeo = b_yeo.locs.values

data1 = data_yeo[data_yeo==1]
xyz1 = xyz_yeo[data_yeo==1]

data2 = data_yeo[data_yeo==2]
xyz2 = xyz_yeo[data_yeo==2]

data3 = data_yeo[data_yeo==3]
xyz3 = xyz_yeo[data_yeo==3]

data4 = data_yeo[data_yeo==4]
xyz4 = xyz_yeo[data_yeo==4]

data5 = data_yeo[data_yeo==5]
xyz5 = xyz_yeo[data_yeo==5]

data6 = data_yeo[data_yeo==6]
xyz6 = xyz_yeo[data_yeo==6]

data7 = data_yeo[data_yeo==7]
xyz7 = xyz_yeo[data_yeo==7]

kwargs = {}
kwargs['alpha'] = 0.2


s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)
s_obj_2 = SourceObj('iEEG', xyz2, data=data2, cmap=cmap)
s_obj_2.color_sources(data=data2)
s_obj_3 = SourceObj('iEEG', xyz3, data=data3, cmap=cmap)
s_obj_3.color_sources(data=data3)
s_obj_4 = SourceObj('iEEG', xyz4, data=data4,cmap=cmap)
s_obj_4.color_sources(data=data4)
s_obj_5 = SourceObj('iEEG', xyz5, data=data5,cmap=cmap)
s_obj_5.color_sources(data=data5)
s_obj_6 = SourceObj('iEEG', xyz6, data=data6, cmap=cmap)
s_obj_6.color_sources(data=data6)
s_obj_7 = SourceObj('iEEG', xyz7, data=data7, cmap=cmap)
s_obj_7.color_sources(data=data7)

s_obj_all = s_obj_1 + s_obj_2 + s_obj_3 + s_obj_4 + s_obj_5 + s_obj_6 + s_obj_7


sc = SceneObj(bgcolor='white', size=(1000, 1000))

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(maxs)))

sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(maxs)))
sc.add_to_subplot(b_obj_proj_left, row=1, col=0, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(maxs)))
sc.add_to_subplot(b_obj_proj_right, row=1, col=1, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(maxs)))
sc.add_to_subplot(b_obj_proj_right, row=0, col=1, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir, 'maxs_networks.png'), transparent=False)


sc = SceneObj(bgcolor='white', size=(1000, 1000))

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(hills)))

sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(hills)))
sc.add_to_subplot(b_obj_proj_left, row=1, col=0, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(hills)))
sc.add_to_subplot(b_obj_proj_right, row=1, col=1, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_all, clim=(1, 7), cmap=cmap, alpha=list(norm_vals(hills)))
sc.add_to_subplot(b_obj_proj_right, row=0, col=1, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir, 'hills_networks.png'), transparent=False)

cmap = "yeo_colors_7_l"
sc = SceneObj(bgcolor='white', size=(1000, 1000))


data1 = b_yeo.get_data().values.ravel()
xyz1 = b_yeo.locs.values

s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
s_obj_1.color_sources(data=data1)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)

b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
b_obj_proj_left.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_left, row=1, col=0, rotate='right', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=1, col=1, rotate='left', use_this_cam=True)

b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
b_obj_proj_right.project_sources(s_obj_1, clim=(1, 7), cmap=cmap)
sc.add_to_subplot(b_obj_proj_right, row=0, col=1, rotate='right', use_this_cam=True)

sc.screenshot(os.path.join(fig_dir, '7_networks.png'), transparent=True)
