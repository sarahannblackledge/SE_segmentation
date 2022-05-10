import numpy as np
import SimpleITK as sitk
import os
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/pythonProject/SE_segmentation')
from downsamplePatient import downsamplePatient

id_nums = [2, 4]
patient_name = 'RMH008'
for j in id_nums:
    id_num = str(j)

    character_num = len(id_num)

    if character_num < 2:
        fpath_mask = "/Users/sblackledge/Documents/Gynae_data_correct/" + patient_name + "/uterus_masks/Uterocervix_0" + id_num + ".nrrd"
    else:
        fpath_mask = "/Users/sblackledge/Documents/Gynae_data_correct/" + patient_name + "/uterus_masks/Uterocervix_" + id_num + ".nrrd"

    CT_dir = "/Users/sblackledge/Documents/Gynae_data_correct/" + patient_name + "/CBCT" + id_num + "_resampled"

    # Make sitk image object
    files_CT = np.array([os.path.join(CT_dir, fl) for fl in os.listdir(CT_dir) if "dcm" in fl])
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels = True) for fl in files_CT])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files_CT = files_CT[np.argsort(locations)]
    CT_sitk = sitk.ReadImage(files_CT)

    # Load in mask
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(fpath_mask)
    uterus_mask = reader.Execute()

    # Downsample to 256x256
    CT_sitk = downsamplePatient(CT_sitk, 2, is_label=False)
    uterus_mask = downsamplePatient(uterus_mask, 2, is_label=True)

    # Extract image arrays from sitk objects
    orig_im = sitk.GetArrayFromImage(CT_sitk)
    mask3D = sitk.GetArrayFromImage(uterus_mask)

    # Normalize intensity values betwen air(-1024) and bone (~1000)
    # Note: opted not to normalize between 0 and 1 explicitly because tattoo markers on some patients have HU of > 4000, so would skew data
    # Chosen approach should approximately normalize between 0 and 1 given that HU of bone can be up to 1000.

    normalized_im3D = (orig_im - np.min(orig_im)) / (1000 - np.min(orig_im))

    # Transpose to conventional slice order
    normalized_im3D = normalized_im3D.transpose(1, 2, 0)
    mask3D = mask3D.transpose(1, 2, 0)

    #Transpose to sagittal orientation
    sag_im3D = normalized_im3D.transpose(2, 0, 1)
    sag_im3D = np.flipud(sag_im3D)

    sag_mask3D = mask3D.transpose(2, 0, 1)
    sag_mask3D = np.flipud(sag_mask3D)

    plt.figure()
    plt.imshow(normalized_im3D[:, :, 65], cmap='gray')
    plt.colorbar()
    plt.clim(0, 0.7)
    plt.contour(mask3D[:, :, 65], 1, colors='m')
    plt.show()

    plt.figure()
    plt.imshow(sag_im3D[:, :, 120], cmap='gray')
    plt.colorbar()
    plt.clim(0, 0.7)
    plt.contour(sag_mask3D[:, :, 120], 1, colors='m')
    plt.show()
