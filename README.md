This repository consist of the code utilized to generate figures and plots used to visualize and analyze results related to my specialization project. 
This ReadMe file holds information on all the python files.
-----------------------------------------------------------
### CountingVoxels.py
Used to find the volumes of the prostate and tumor masks provided as binary masks. 
Output provides number of voxels with intensities above a threshold, as well as voxel dimensions (x,y,z) in mm length. 
The commented out section also provides the longest cross measure in the axial plane. 
Results were used as a basid for volumetric analysis of prostate volume development and response to treatment.

### CountVoxConfig.yaml
Consist of the unput path to the binary mask being analyzed, used by 'CountingVoxels.py'.

### CroppingImages.py
Used to crop images that were too large (too large FOV) for the AI tool to handle. 
Center coordinates and FOV size (in x,y,z direction) had to be predetermined. 
As much metadata and other information about the image slices were saved.
Output (the cropped images) were saved in a designated output folder to be further analyzed.

### CroppingImConfig.yaml
Consist of input file path, output folder path, as well as center coordinates and cropping size.
Utilized by 'CroppingImages.py'

### BinMaskOnc.py
Used to create a binary mask from RT-STRUCT delination provided by oncologist. 
An rt-struct file, reference t2w image are needed as input.
Also center of cropping and cropping size is used to match the size of the new created binary mask to that segmented by the AI tool.
Output (the binary mask) were saved in a designated folder for further analysis.

### 3DVisualization.py
This code is used to visualize multiple volumes/binary mask in comparison to each other.
The first part of the code, the commented out section by default, is used to visualize both the segmented prostate and the delineated prostate. These needed to be centered around a central axis so they could be compared.
The second part of the code is used to visualize the prostate and any tumor lesion found inside. These were not centered as the location of the tumor is of importance. 
For both code parts, the figure from two viewpoints (axial and coronal) were saved in a designated output folder.

-------------------------------------------------------------------------------

Some portions of the code used in this work were generated with the assistance of OpenAI's ChatGPT, a large language model, and subsequently adapted for this study.
Ref: OpenAI. ChatGPT (December 2024 version). 2024. OpenAI, https://openai.com/chatgpt.


