
import supereeg as se
import os
import numpy as np
import glob as glob
from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj

colors = 'RdBu_r'

nii_bo_dir = '../../../data/niis/pcas'
fig_dir = '../../../paper/figs/source/pcas'

r = 10

CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='black', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)

template_brain = 'B3'

conditions = ['intact', 'paragraph', 'rest', 'word']

for e, c in enumerate(conditions):

    cmap = colors

    b_1 = se.load(os.path.join(nii_bo_dir,f'{c}_1_pca_chunk_smallest_change3.bo'))
    b1 = se.load(os.path.join(nii_bo_dir, f'{c}_1_pca_chunk_largest_change3.bo'))
    b_2 = se.load(os.path.join(nii_bo_dir,f'{c}_2_pca_chunk_smallest_change3.bo'))
    b2 = se.load(os.path.join(nii_bo_dir, f'{c}_2_pca_chunk_largest_change3.bo'))
    b_3 = se.load(os.path.join(nii_bo_dir,f'{c}_3_pca_chunk_smallest_change3.bo'))
    b3 = se.load(os.path.join(nii_bo_dir, f'{c}_3_pca_chunk_largest_change3.bo'))
    b_4 = se.load(os.path.join(nii_bo_dir,f'{c}_4_pca_chunk_smallest_change3.bo'))
    b4 = se.load(os.path.join(nii_bo_dir, f'{c}_4_pca_chunk_largest_change3.bo'))
    b_5= se.load(os.path.join(nii_bo_dir,f'{c}_5_pca_chunk_smallest_change3.bo'))
    b5 = se.load(os.path.join(nii_bo_dir, f'{c}_5_pca_chunk_largest_change3.bo'))

    data1 = b1.get_data().values.ravel() * .9
    xyz1 = b1.locs.values

    data2 = b2.get_data().values.ravel() * .7
    xyz2 = b2.locs.values

    data3 = b3.get_data().values.ravel() * .5
    xyz3 = b3.locs.values

    data4 = b4.get_data().values.ravel() * .3
    xyz4 = b4.locs.values

    data5 = b5.get_data().values.ravel() * .1
    xyz5 = b5.locs.values

    data_1 = b_1.get_data().values.ravel() * -.9
    xyz_1 = b_1.locs.values

    data_2 = b_2.get_data().values.ravel() * -.7
    xyz_2 = b_2.locs.values

    data_3 = b_3.get_data().values.ravel() * -.5
    xyz_3 = b_3.locs.values

    data_4 = b_4.get_data().values.ravel() * -.3
    xyz_4 = b_4.locs.values

    data_5 = b_5.get_data().values.ravel() * -.1
    xyz_5 = b_5.locs.values

    template_brain = 'B3'

    sc = SceneObj(bgcolor='white', size=(1000, 1000))

    CBAR_STATE = dict(cbtxtsz=12, clim=[-1.5, 1.5], txtsz=10., width=.1, cbtxtsh=3.,
                      rect=(-.3, -2., 1., 4.))
    KW = dict(title_size=14., zoom=1)

    s_obj_1 = SourceObj('iEEG', xyz1, data=data1, cmap=cmap)
    s_obj_1.color_sources(data=data1)
    s_obj_2 = SourceObj('iEEG', xyz2, data=data2, cmap=cmap)
    s_obj_2.color_sources(data=data2)
    s_obj_3 = SourceObj('iEEG', xyz3, data=data3, cmap=cmap)
    s_obj_3.color_sources(data=data3)
    s_obj_4 = SourceObj('iEEG', xyz4, data=data4, cmap=cmap)
    s_obj_4.color_sources(data=data4)
    s_obj_5 = SourceObj('iEEG', xyz5, data=data5, cmap=cmap)
    s_obj_5.color_sources(data=data5)

    s_obj_1_ = SourceObj('iEEG', xyz_1, data=data_1, cmap=cmap)
    s_obj_1_.color_sources(data=data_1)
    s_obj_2_ = SourceObj('iEEG', xyz_2, data=data_2, cmap=cmap)
    s_obj_2_.color_sources(data=data_2)
    s_obj_3_ = SourceObj('iEEG', xyz_3, data=data_3, cmap=cmap)
    s_obj_3_.color_sources(data=data_3)
    s_obj_4_ = SourceObj('iEEG', xyz_4, data=data_4, cmap=cmap)
    s_obj_4_.color_sources(data=data_4)
    s_obj_5_ = SourceObj('iEEG', xyz_5, data=data_5, cmap=cmap)
    s_obj_5_.color_sources(data=data_5)

    #s_obj_all = s_obj_1_ + s_obj_2_ + s_obj_3_ + s_obj_4_ + s_obj_5_

    #s_obj_all = s_obj_1 + s_obj_2 + s_obj_3 + s_obj_4 + s_obj_5
    s_obj_all = s_obj_1 + s_obj_2+ s_obj_3+ s_obj_4 + s_obj_5 + s_obj_1_ + s_obj_2_+ s_obj_3_+ s_obj_4_ + s_obj_5_

    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_all, clim=(-1.5, 1.5), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)


    b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
    b_obj_proj_left.project_sources(s_obj_all, clim=(-1.5, 1.5), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_all, clim=(-1.5, 1.5), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

    b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
    b_obj_proj_right.project_sources(s_obj_all, clim=(-1.5, 1.5), cmap=cmap)
    sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

    sc.screenshot(os.path.join(fig_dir, f'{c}_plot_changes.png'), transparent=True)