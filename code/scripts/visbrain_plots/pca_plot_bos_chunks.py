
import supereeg as se
import os
import glob as glob
from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj

colors = ['Blues_r', 'Greens_r', 'Purples_r', 'Reds_r']
#cmap = "Purples_r"

nii_bo_dir = '../../../data/niis/pcas'
fig_dir = '../../../paper/figs/source/pcas'

r = 10

CBAR_STATE = dict(cbtxtsz=20, txtsz=20., txtcolor='black', width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))
KW = dict(title_size=14., zoom=2)

template_brain = 'B3'

conditions = ['intact', 'paragraph', 'rest', 'word']

chunks = 3


for chunk in range(chunks):
    for e, c in enumerate(conditions):

        cmap = colors[e]

        b1 = se.load(os.path.join(nii_bo_dir,f'{c}_1_pca_chunk_{chunk+1}.bo'))
        b2 = se.load(os.path.join(nii_bo_dir,f'{c}_2_pca_chunk_{chunk+1}.bo'))
        b3 = se.load(os.path.join(nii_bo_dir,f'{c}_3_pca_chunk_{chunk+1}.bo'))
        b4 = se.load(os.path.join(nii_bo_dir,f'{c}_4_pca_chunk_{chunk+1}.bo'))
        b5 = se.load(os.path.join(nii_bo_dir,f'{c}_5_pca_chunk_{chunk+1}.bo'))

        data1 = b1.get_data().values.ravel()
        xyz1 = b1.locs.values

        data2 = b2.get_data().values.ravel()
        xyz2 = b2.locs.values

        data3 = b3.get_data().values.ravel()
        xyz3 = b3.locs.values

        data4 = b4.get_data().values.ravel()
        xyz4 = b4.locs.values

        data5 = b5.get_data().values.ravel()
        xyz5 = b5.locs.values


        template_brain = 'B3'

        sc = SceneObj(bgcolor='white', size=(1000, 1000))

        CBAR_STATE = dict(cbtxtsz=12, clim=[0, 6], txtsz=10., width=.1, cbtxtsh=3.,
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

        s_obj_all = s_obj_1 + s_obj_2 + s_obj_3 + s_obj_4 + s_obj_5


        b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
        b_obj_proj_left.project_sources(s_obj_all, clim=(0, 4), cmap=cmap)
        sc.add_to_subplot(b_obj_proj_left, row=0, col=0, rotate='left', use_this_cam=True)


        b_obj_proj_left = BrainObj(template_brain, hemisphere='left', translucent=False)
        b_obj_proj_left.project_sources(s_obj_all, clim=(0, 4), cmap=cmap)
        sc.add_to_subplot(b_obj_proj_left, row=0, col=1, rotate='right', use_this_cam=True)

        b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
        b_obj_proj_right.project_sources(s_obj_all, clim=(0, 4), cmap=cmap)
        sc.add_to_subplot(b_obj_proj_right, row=0, col=2, rotate='left', use_this_cam=True)

        b_obj_proj_right = BrainObj(template_brain, hemisphere='right', translucent=False)
        b_obj_proj_right.project_sources(s_obj_all, clim=(0, 4), cmap=cmap)
        sc.add_to_subplot(b_obj_proj_right, row=0, col=3, rotate='right', use_this_cam=True)

        sc.screenshot(os.path.join(fig_dir, f'{c}_5_components_chunk_{chunk+1}.png'), transparent=True)