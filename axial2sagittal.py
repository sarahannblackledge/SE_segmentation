#Read in 2D dicom files in axial orientation and convert into directory of 2D sagittal numpy arrays
import numpy as np
import SimpleITK as sitk
import os
import pydicom as dicom
import matplotlib.pyplot as plt


def axial2sagittal(CT_dir, fpath_uterus_mask):
    # Make sitk image object
    files_CT = np.array([os.path.join(CT_dir, fl) for fl in os.listdir(CT_dir) if "dcm" in fl])
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels = True) for fl in files_CT])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files_CT = files_CT[np.argsort(locations)]
    CT = sitk.ReadImage(files_CT)

    #reorient dicom and save as nrrd
    '''reoriented = sitk.DICOMOrient(CT, 'SPR')
    #sitk.WriteImage(reoriented, '/Users/sblackledge/Desktop/SPR.nrrd')'''

    #reorient numpy array
    CT_im = sitk.GetArrayFromImage(CT)
    axial_orientation = CT_im.transpose(1, 2, 0)
    sagittal_orientation = axial_orientation.transpose(2, 0, 1)
    sagittal_orientation = np.flipud(sagittal_orientation)

    #Load in uterus npy mask
    ax_label = np.load(fpath_uterus_mask)
    ax_label = ax_label.transpose(1,0, 2)
    sag_label = ax_label.transpose(2, 0, 1)
    sag_label = np.flipud(sag_label)

    plt.figure(1)
    plt.imshow(axial_orientation[:, :, 113], cmap='gray')
    plt.contour(ax_label[:, :, 113], 1, colors='m')
    plt.show()

    plt.figure(2)
    plt.imshow(sagittal_orientation[:, :, 250], cmap='gray')
    plt.contour(sag_label[:, :, 250], 1, colors='m')
    plt.show()


    return axial_orientation, sagittal_orientation

CT_dir = "/Users/sblackledge/Documents/Gynae_data_correct/Gynae1_1025/CBCT5_resampled"
fpath_uterus_mask = "/Users/sblackledge/Documents/Gynae_data_correct/Gynae1_1025/uterus_masks/Uterocervix_05.npy"
axial_orientation, sagittal_orientation = axial2sagittal(CT_dir, fpath_uterus_mask)





