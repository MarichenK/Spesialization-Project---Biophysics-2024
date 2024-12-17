import numpy as np
import pydicom
import os
import nibabel as nib
from skimage.draw import polygon #Fill contours in the mask
import SimpleITK as sitk

def load_dicom_volume(dicom_folder):
    
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
    dicom_slices = [pydicom.dcmread(f) for f in dicom_files]
    
    #Sort slices by their Image Position Patient (z-coordinate) to align in 3D space
    dicom_slices.sort(key=lambda s: s.ImagePositionPatient[2])
    volume = np.stack([s.pixel_array for s in dicom_slices], axis=-1)
    
    return volume, dicom_slices

#Retrieve contour/delineation from rt-struct file
def get_contours(rtstruct):
    contours = []
    for roi_contour in rtstruct.ROIContourSequence:
        for contour_sequence in roi_contour.ContourSequence:
            contour_data = contour_sequence.ContourData
            num_points = len(contour_data) // 3
            contour_points = np.array(contour_data).reshape((num_points, 3))
            contours.append((contour_points, contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID))
    return contours


#Convert to pixels in order to compare and use for further analysis
def convert_to_pixel(contour_points, dicom_slice):
    pixel_spacing = dicom_slice.PixelSpacing
    image_position = dicom_slice.ImagePositionPatient
    row_spacing, col_spacing = pixel_spacing[1], pixel_spacing[0]

    col_coords = (contour_points[:, 0] - image_position[0]) / col_spacing
    row_coords = (contour_points[:, 1] - image_position[1]) / row_spacing
    
    col_coords = np.round(col_coords).astype(int)
    row_coords = np.round(row_coords).astype(int)

    return row_coords, col_coords


#Creating a binary mask so that this can be compared to segmentated volumes, and be visualized in other code. 
#We need cropping info so that it matches what it is being compared to
def create_bin_mask(rtstruct_path, dicom_folder, center, crop_dims, output_path):

    rtstruct = pydicom.dcmread(rtstruct_path)
    reference_volume, dicom_slices = load_dicom_volume(dicom_folder)
    cols, rows, num_slices = reference_volume.shape
    mask = np.zeros((cols, rows, num_slices), dtype=np.uint8)

    contours = get_contours(rtstruct)
    
    for contour_points, sop_instance_uid in contours:
        slice_index = next(i for i, s in enumerate(dicom_slices) if s.SOPInstanceUID == sop_instance_uid)
        dicom_slice = dicom_slices[slice_index]

        row_coords, col_coords = convert_to_pixel(contour_points, dicom_slice)
      
        rr, cc = polygon(row_coords, col_coords, shape=mask[:, :, slice_index].shape)
        mask[rr, cc, slice_index] = 1 


    mask[mask > 0] = 1

    x_center, y_center, z_center = center
    dx, dy, dz = crop_dims

    x_start = x_center - (dx//2)
    x_end = x_center + (dx//2)
    y_start = y_center - (dy//2)
    y_end = y_center + (dy//2)
    z_start = z_center - (dz//2)
    z_end = z_center + (dz//2)

    cropped_mask = mask[x_start:x_end, y_start:y_end, z_start:z_end]

    os.makedirs(output_path, exist_ok=True)

    nifti_img = nib.Nifti1Image(cropped_mask, affine=np.eye(4))
    output_file = os.path.join(output_path, 'cropped_mask441.nii.gz')
    nib.save(nifti_img, output_file)
    print('Cropping complete, saved in file')


