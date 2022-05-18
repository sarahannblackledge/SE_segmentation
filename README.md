# SE_segmentation
Fully automated stacked-ellipse segmentation of uterocervix on CBCT.

STEP 1: Preprocess data:
    Crop and resample image data so format conducive for AI-based segmentation
    run 'preprocess_data.py', which calls 'crop_sitk_im.py' and 'resample_patient.py'
    OUTPUTs:
    (1) nrrd files of cropped, resampled, and intensity normalized images and corresponding masks
    in patient subdirectories 'processed_CBCTs_nrrd' and 'processed_masks_nrrd', respectively.
    (2) png files of cropped, resampled, and intensity normalized images in 2D sagittal orientation
    and corresponding labels in 'sagittal_data_CBCT' directory


Segmentation of uterus on sagittal plane. Only interested in central-most sagittal slices
