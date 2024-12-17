
import SimpleITK as sitk
import os
import yaml

def load_config(yaml_file):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file) 
    return config

config = load_config('/mnt/work/users/marichek/Cropping_config.yml') 

input_path = config['paths']['input_path']
output_path = config['paths']['output_path']
center = config['settings']['center_coordinates'] #3D coord
crop_size = config['settings']['crop_size'] #3D coord

start = [center[i] - crop_size[i] // 2 for i in range(3)] 

#########################################################################
#Common for both DICOM and nii

def crop_image(image, start, crop_size):
    return sitk.RegionOfInterest(image, crop_size, start)

def save_3d_image_as_dicom(cropped_image, output_folder, dicom_filenames, first_slice_metadata, start_z):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Extracting the number of slices in the cropped image
    num_slices = cropped_image.GetSize()[2]

    #Getting the filenames for the corresponding original slices
    remaining_filenames = dicom_filenames[start_z:start_z+num_slices]

    if num_slices != len(remaining_filenames):
        raise ValueError(f"Mismatch between number of slices ({num_slices}) and DICOM files ({len(dicom_filenames)})")

    #Extracting original DICOM metadata such as direction, origin, spacing
    original_direction = first_slice_metadata['direction']
    original_origin = first_slice_metadata['origin']
    original_spacing = first_slice_metadata['spacing']

    #Apply original metadata to the cropped image
    cropped_image.SetDirection(original_direction)
    cropped_image.SetOrigin(original_origin)
    cropped_image.SetSpacing(original_spacing)

    series_writer = sitk.ImageFileWriter()
    series_writer.KeepOriginalImageUIDOn()

    extractor = sitk.ExtractImageFilter()

    #Saving each slice in the cropped 3D image as a separate DICOM file
    for i in range(num_slices):
        #Extracting 2D slice from the cropped 3D
        slice_index = [0,0,i]
        size = [cropped_image.GetSize()[0], cropped_image.GetSize()[1], 0]
        

        extractor.SetSize(size)
        extractor.SetIndex(slice_index)
        slice_2d = extractor.Execute(cropped_image)

        slice_position = list(original_origin)
        slice_position[2] = original_origin[2] + (start_z + i) * original_spacing[2]

        #Add metadata to the DICOM slice
        #Uncertain how much of this metadata is used when running Proviz, so we try to save as much as possible
        slice_2d.SetMetaData('0020|000D', first_slice_metadata.get('0020|000D', "Unknown"))  
        slice_2d.SetMetaData('0020|000E', first_slice_metadata.get('0020|000E', "Unknown"))  
        slice_2d.SetMetaData('0020|0032', '\\'.join(map(str, slice_position))) 
        slice_2d.SetMetaData('0020|0013', str(i + 1))  #Instance number (slice number)

        original_filename = os.path.basename(remaining_filenames[i])
        output_filepath = os.path.join(output_folder, os.path.splitext(original_filename)[0] + '.dcm')

        #Saving the slices as DICOM, compatible with Proviz
        series_writer.SetFileName(output_filepath)
        series_writer.Execute(slice_2d)

        print(f"Saved slice {i + 1} to {output_filepath}")


#For DICOM
def process_dicom_series(input_folder, output_folder, start, crop_size):
    """Process the DICOM series, crop the image, and save the result as a 3D DICOM series."""

    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_filenames = dicom_series_reader.GetGDCMSeriesFileNames(input_folder)
    dicom_series_reader.SetFileNames(dicom_filenames)

    image_3d = dicom_series_reader.Execute()

   
    if image_3d is None:
        raise RuntimeError("Failed to read DICOM series.")

    first_slice_image = sitk.ReadImage(dicom_filenames[0])  

    '''print("Available metadata keys:")
    for key in first_slice_image.GetMetaDataKeys():
        print(key)'''

    #Extracting metadata with a fallback if the keys do not exist
    first_slice_metadata = {
        '0020|000D': first_slice_image.GetMetaData('0020|000D') if first_slice_image.HasMetaDataKey('0020|000D') else "Unknown",  # StudyInstanceUID
        '0020|000E': first_slice_image.GetMetaData('0020|000E') if first_slice_image.HasMetaDataKey('0020|000E') else "Unknown",  # SeriesInstanceUID
        'direction': image_3d.GetDirection(),
        'origin': image_3d.GetOrigin(),
        'spacing': image_3d.GetSpacing(),
    }

    cropped_image = crop_image(image_3d, start, crop_size)
    start_z = start[2]
    save_3d_image_as_dicom(cropped_image, output_folder, dicom_filenames, first_slice_metadata, start_z)

