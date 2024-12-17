

#################################### This code is for 3D visualization of Proviz segment and drawn prostate from oncologist #########################################################
'''
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import binary_fill_holes

def plot_3d_onc_seg(oncology_image_path, segment_image_path, oncology_threshold=0.5, segment_threshold=0.5, save_path_prefix=None):
    oncology_image = nib.load(oncology_image_path)
    segment_image = nib.load(segment_image_path)

    #Get image data as numpy arrays
    oncology_data = oncology_image.get_fdata()
    segment_data = segment_image.get_fdata()

    #x and y axis has been switched, need to rotate this plane. 
    oncology_data = np.rot90(oncology_data, k=-1, axes=(0,1))

    #Affine matrices and voxel sizes
    oncology_affine = oncology_image.affine
    segment_affine = segment_image.affine

    #Binary masks
    oncology_mask = oncology_data > oncology_threshold
    segment_mask = segment_data > segment_threshold

    #Transform voxel indices to world coordinates (in mm)
    oncology_voxels = np.argwhere(oncology_mask)
    segment_voxels = np.argwhere(segment_mask)
    oncology_coords = nib.affines.apply_affine(oncology_affine, oncology_voxels)
    segment_coords = nib.affines.apply_affine(segment_affine, segment_voxels)

    #Calculate centroids in real-world coordinates (mm)
    oncology_centroid = oncology_coords.mean(axis=0)
    segment_centroid = segment_coords.mean(axis=0)

    #Calculate offset to align segment to oncology centroid in mm
    offset = oncology_centroid - segment_centroid
    segment_coords_aligned = segment_coords + offset

    #Apply offset to align segment mask
    segment_mask_aligned = np.zeros_like(oncology_mask, dtype=bool)
    aligned_indices = np.round(nib.affines.apply_affine(np.linalg.inv(oncology_affine), segment_coords_aligned)).astype(int)
    valid_indices = (aligned_indices[:, 0] < oncology_mask.shape[0]) & (aligned_indices[:, 1] < oncology_mask.shape[1]) & (aligned_indices[:, 2] < oncology_mask.shape[2])
    segment_mask_aligned[aligned_indices[valid_indices, 0], aligned_indices[valid_indices, 1], aligned_indices[valid_indices, 2]] = True

    #Generate surfaces with Marching Cubes
    oncology_verts, oncology_faces, _, _ = measure.marching_cubes(oncology_mask, level=0)
    segment_verts, segment_faces, _, _ = measure.marching_cubes(segment_mask_aligned, level=0)

    #Convert vertices to real-world coordinates
    oncology_verts = nib.affines.apply_affine(oncology_affine, oncology_verts)
    segment_verts = nib.affines.apply_affine(oncology_affine, segment_verts) #!!!!

    #Define viewpoints
    viewpoints = [
        (30, 30),   # Front view
        (30, 150),  # Side view
        (30, 270),  # Another side view
        (90, 0),    # Top view
        (0,90),     #Saggital
        (45,45)     #Coronal
    ]

    for elev, azim in viewpoints:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        #Create surface plots for Oncology and Segment
        oncology_mesh = Poly3DCollection(oncology_verts[oncology_faces], alpha=0.4, facecolor='sandybrown')
        segment_mesh = Poly3DCollection(segment_verts[segment_faces], alpha=0.5, facecolor='cornflowerblue')

        ax.add_collection3d(oncology_mesh)
        ax.add_collection3d(segment_mesh)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        ax.set_xlim(oncology_centroid[0] - 30, oncology_centroid[0] + 30)
        ax.set_ylim(oncology_centroid[1] - 30, oncology_centroid[1] + 30)
        ax.set_zlim(oncology_centroid[2] - 30, oncology_centroid[2] + 30)


        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        ax.set_title("3D Surface Plot of Prostate drawn by oncologist and Proviz (Aligned by Centroids)")
        ax.view_init(elev, azim)

        
        if save_path_prefix:
            save_path = f"{save_path_prefix}_elev{elev}_azim{azim}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {save_path}")

        plt.show()
        plt.close(fig)

'''




############################################# This code is for 3D visualize prostate and tumor #############################################################

import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_3d_prostate_tumor(prostate_image_path, tumor_image_path, prostate_threshold=0.5, tumor_threshold=0.5, save_path_prefix=None):
    #Loading the NIfTI images
    prostate_image = nib.load(prostate_image_path)
    tumor_image = nib.load(tumor_image_path)

    #Get the image data as numpy arrays, affine matrices (need this because voxel size and 'real world' coordinates are not the same)
    prostate_data = prostate_image.get_fdata()
    tumor_data = tumor_image.get_fdata()

    prostate_affine = prostate_image.affine
    tumor_affine = tumor_image.affine

    #Create binary masks for prostate and tumor volumes
    prostate_mask = prostate_data > prostate_threshold
    tumor_mask = tumor_data > tumor_threshold

    #Voxels
    prostate_voxels = np.argwhere(prostate_mask)
    tumor_voxels = np.argwhere(tumor_mask)
    prostate_coords = nib.affines.apply_affine(prostate_affine, prostate_voxels)
    tumor_coords = nib.affines.apply_affine(tumor_affine, tumor_voxels)

    #Making box for visualization
    bbox_min = np.min(prostate_coords, axis = 0)
    bbox_max = np.max(prostate_coords, axis = 0)
    bbox_center = (bbox_min + bbox_max) // 2

    grid_center = np.array([30, 30, 30])

    shift = grid_center - bbox_center

    shifted_prostate_coords = prostate_coords + shift
    shifted_tumor_coords = tumor_coords + shift

    grid_shape = (60, 60, 60)
    prostate_voxel_grid = np.zeros(grid_shape, dtype=bool)
    tumor_voxel_grid = np.zeros(grid_shape, dtype=bool)

    def map_to_grid(coords):
        grid_coords = np.round(coords).astype(int)
        mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < grid_shape[0]) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < grid_shape[1]) &
            (grid_coords[:, 2] >= 0) & (grid_coords[:, 2] < grid_shape[2])
        )
        return grid_coords[mask]

    prostate_voxel_indices = map_to_grid(shifted_prostate_coords)
    tumor_voxel_indices = map_to_grid(shifted_tumor_coords)

    prostate_voxel_grid[
        prostate_voxel_indices[:, 0],
        prostate_voxel_indices[:, 1],
        prostate_voxel_indices[:, 2]
    ] = True

    tumor_voxel_grid[
        tumor_voxel_indices[:, 0],
        tumor_voxel_indices[:, 1],
        tumor_voxel_indices[:, 2]
    ] = True


    #Different viewing angles (elevation, azimuth)
    viewpoints = [
        (90, 0),   #Axial view
        (0,90)    #Coronal view
    ]

    axis_range = (0,60)

    for elev, azim in viewpoints:
        #Create a new figure for each viewpoint
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.voxels(prostate_voxel_grid, facecolors='salmon', edgecolor='none', alpha=0.2)
        ax.voxels(tumor_voxel_grid, facecolors='royalblue', edgecolor='none', alpha=0.9)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        ax.set_xlim(0,60)
        ax.set_ylim(0,60)
        ax.set_zlim(0,60)

        ax.set_box_aspect([1,1,1])

        if (elev, azim) == (90,0):
            ax.invert_xaxis()
            ax.invert_zaxis()
            ax.view_init(90,270)
        else:
            ax.view_init(elev, azim)

        ax.set_title("3D Plot of Prostate Gland (red) and Tumor Lesion (blue)")

        if save_path_prefix:
            save_path = f"{save_path_prefix}_elev{elev}_azim{azim}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  
            print(f"Figure saved as {save_path}")
        
        plt.show()
        plt.close(fig)



