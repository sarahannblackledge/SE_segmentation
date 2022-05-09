#Read in 2D dicom files in axial orientation and convert into directory of 2D sagittal numpy arrays
import numpy as np
import SimpleITK as sitk
import os
import pydicom as dicom
import matplotlib.pyplot as plt


def axial2sagittal(CT_dir):
    # Make sitk image object
    files_CT = np.array([os.path.join(CT_dir, fl) for fl in os.listdir(CT_dir) if "dcm" in fl])
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels = True) for fl in files_CT])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files_CT = files_CT[np.argsort(locations)]
    CT = sitk.ReadImage(files_CT)

    #Get image data
    image_out = sitk.GetImageFromArray(sitk.GetArrayFromImage(CT))

    #Set up other image characteristics
    image_out.SetOrigin(CT.GetOrigin())
    image_out.SetSpacing(CT.GetSpacing())

    ''''#Set to sagittal
    directioncosines = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    image_out.SetDirection(directioncosines)
    #sitk.WriteImage(image_out, '/Users/sblackledge/Desktop/test.nrrd')'''

    image_out = sitk.DICOMOrient(CT, 'PSL')
    sitk.WriteImage(image_out, '/Users/sblackledge/Desktop/test.nrrd')
    return CT, image_out

CT_dir = "/Users/sblackledge/Documents/Gynae_data_correct/Gynae1_1025/CBCT3_resampled"

CT, image_out = axial2sagittal(CT_dir)

#Display
plt.figure(0)
plt.title('original image')
CT_im = sitk.GetArrayFromImage(CT)
plt.imshow(CT_im[75], cmap='gray')
plt.show()

plt.figure(1)
plt.title('re-oriented image')
CT_im2 = sitk.GetArrayFromImage(image_out)
plt.imshow(CT_im2[175], cmap='gray')
plt.show()
