import numpy as np
import SimpleITK as sitk
import os
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/sblackledge/PycharmProjects/pythonProject/SE_segmentation')
from crop_sitk_im import crop_sitk_im
from resample_patient import resample_patient
import cv2

#Prepares data for semantic segmentation using U-Net. Specifically, 3D data is cropped, resampled, and intensity-normalized
#outputs:
# (1) nrrd files of image and corresponding mask in patient subdirectories 'processed_CBCTs_nrrd' and 'processesd_masks_nrrd', respectivley
# (2) png files of 2D sagittal images and corresponding mask in directory 'sagittal_data_CBCT'

#This is not actually a function, so manually edit the 'id_nums" and 'patient_name' variables in-line


#plotflag 0 to hide plots, any other number to show
plotflag = 1

id_nums = [2]
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
    dicoms = np.array([dicom.read_file(fl, stop_before_pixels=True) for fl in files_CT])
    locations = np.array([float(dcm.ImagePositionPatient[-1]) for dcm in dicoms])
    files_CT = files_CT[np.argsort(locations)]
    CT_sitk = sitk.ReadImage(files_CT)
    # Load in mask
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(fpath_mask)
    uterus_mask = reader.Execute()

    #Crop sitk image and corresponding mask to remove background pixels
    cropped_img, cropped_mask = crop_sitk_im(CT_sitk, uterus_mask)

    # Downsample to 256x256x130
    desired_dimensions = [256, 256, 256]
    CT_sitk = resample_patient(cropped_img, desired_dimensions, is_label=False)
    uterus_mask = resample_patient(cropped_mask, desired_dimensions, is_label=True)

    # Extract image arrays from sitk objects
    orig_im = sitk.GetArrayFromImage(CT_sitk)
    mask3D = sitk.GetArrayFromImage(uterus_mask)

    # Normalize intensity values betwen air(-1024) and bone (~1000)
    # Note: opted not to normalize between 0 and 1 explicitly because tattoo markers on some patients have HU of > 4000, so would skew data
    # Chosen approach should approximately normalize between 0 and 1 given that HU of bone can be up to 1000.

    normalized_im3D = (orig_im - np.min(orig_im)) / (1000 - np.min(orig_im))
    #-------------------------------------------------------------------------
    #Write to nrrd file
    CT_sitk_normalized = sitk.GetImageFromArray(normalized_im3D)
    CT_sitk_normalized.CopyInformation(CT_sitk)

    mydir = "/Users/sblackledge/Documents/Gynae_data_correct/" + patient_name + "/processed_CBCTs_nrrd"
    mydir2 = "/Users/sblackledge/Documents/Gynae_data_correct/" + patient_name + "/processed_masks_nrrd"
    check_folder = os.path.isdir(mydir)
    check_folder2 = os.path.isdir(mydir2)

    if not check_folder:
        os.makedirs(mydir)

    if not check_folder2:
        os.makedirs(mydir2)

    fname_im_nrrd = 'CBCT'+id_num + ".nrrd"
    fname_mask_nrrd = 'CBCT'+id_num + ".nrrd"

    fpath_im_nrrd = os.path.join(mydir, fname_im_nrrd)
    fpath_mask_nrrd = os.path.join(mydir2, fname_mask_nrrd)

    sitk.WriteImage(CT_sitk_normalized, fpath_im_nrrd)
    sitk.WriteImage(uterus_mask, fpath_mask_nrrd)
    #----------------------------------------------------------------------
    # Transpose to conventional slice order
    normalized_im3D = normalized_im3D.transpose(1, 2, 0)
    mask3D = mask3D.transpose(1, 2, 0)

    #Transpose to sagittal orientation
    sag_im3D = normalized_im3D.transpose(2, 0, 1)
    sag_im3D = np.flipud(sag_im3D)

    sag_mask3D = mask3D.transpose(2, 0, 1)
    sag_mask3D = np.flipud(sag_mask3D)

    #Save each non-zero slice as individual npy array in 'sagittal_data_CBCT' folder
    img_path = '/Users/sblackledge/Documents/Gynae_data_correct/sagittal_data_CBCT/images'
    label_path = '/Users/sblackledge/Documents/Gynae_data_correct/sagittal_data_CBCT/labels'

    arr_size = sag_mask3D.shape
    for i in range(0, arr_size[2]-1):
        sag_slice = (sag_im3D[:, :, i])*255
        label_slice = (sag_mask3D[:, :, i])*255

        fname = patient_name + "CBCT" + id_num + "_" + str(i)+'.png'

        fpath_img = os.path.join(img_path, fname)
        fpath_label = os.path.join(label_path, fname)

        #Save as np arrays
        '''np.save(fpath_img, sag_slice)
        np.save(fpath_label, label_slice)'''

        #Save as png
        #plt.imsave(fpath_img, sag_slice, cmap='gray', vmin=0, vmax=.7)
        #plt.imsave(fpath_label, label_slice, cmap='gray')

        sag_slice = sag_slice.astype('uint8')
        label_slice = label_slice.astype('uint8')

        cv2.imwrite(fpath_img, sag_slice)
        cv2.imwrite(fpath_label, label_slice)



    #Visualize example slices for sanity check (optional: hardcode plotflag variable to show/hide)
    if plotflag != 0:
        '''plt.figure()
        plt.imshow(normalized_im3D[:, :, 65], cmap='gray')
        plt.colorbar()
        plt.clim(0, 0.7)
        plt.contour(mask3D[:, :, 65], 1, colors='m')
        plt.show()'''

        plt.figure()
        plt.imshow(sag_im3D[:, :, 125], cmap='gray')
        plt.colorbar()
        plt.clim(0, 0.7)
        plt.contour(sag_mask3D[:, :, 125], 1, colors='m')
        plt.show()

