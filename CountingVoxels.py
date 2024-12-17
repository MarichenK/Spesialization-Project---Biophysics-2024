#Finding volumes

import os
import yaml
import scipy.spatial
import numpy as np

def load_config(yaml_file):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('/mnt/work/users/marichek/CountV_config.yml') 

input_path = config['paths']['input_path']

#Calculating num of voxels
import SimpleITK as sitk

image = sitk.ReadImage(input_path)
image_array = sitk.GetArrayFromImage(image)

count_vol = (image_array > 0.5).sum()
voxel_dimensions = image.GetSpacing()

#Finding longest cross measure
'''
from skimage.measure import find_contours

global_longest_cross_measure = 0

for slice_index in range(image_array.shape[2]):
    slice_2D = image_array[:, :, slice_index]
    contours = find_contours(slice_2D, level = 0.5)
out
    boundary_points = []

    for contour in contours:
        for point in contour:
            boundary_points.append([point[0], point[1]])

    if not boundary_points:
        continue

    boundary_points = np.array(boundary_points)

    boundary_points_physical = boundary_points * np.array([voxel_dimensions[0], voxel_dimensions[1]])

    distance_matrix = scipy.spatial.distance.cdist(boundary_points_physical, boundary_points_physical)
    slice_longest_cross_measure = np.max(distance_matrix)

    if slice_longest_cross_measure > global_longest_cross_measure:
        global_longest_cross_measure = slice_longest_cross_measure

print(f'The longest cross measure is: {global_longest_cross_measure:.2f} mm')'''

print(f"Number of voxels with intensity > 0.5: {count_vol}")
print(f"Voxel dimensions (spacing): {voxel_dimensions}") #should be in mm.

#The input files will contain mask of either prostate gland or tumor lesion. 
#The mask is one pixel/voxel value (hopefully > 50%) and surrounding tissue is another (normally 0). 