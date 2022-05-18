import SimpleITK as sitk

def crop_sitk_im(sitk_im, sitk_mask):

    #background value to crop
    bg_value = -1024

    #Create mask image that is just non-background pixels
    fg_mask = (sitk_im != bg_value)

    #Compute shape statistics on the mask
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(fg_mask)

    # Get the bounds of the mask.
    # Bounds are given as [Xstart, Ystart, Zstart, Xwidth, Ywidth, Zwidth]
    bounds = lsif.GetBoundingBox(1)

    Xmin_crop = bounds[0]
    Ymin_crop = bounds[1]
    Zmin_crop = bounds[2]

    Xmax_crop = sitk_im.GetWidth() - (bounds[0]+bounds[3])
    Ymax_crop = sitk_im.GetHeight() - (bounds[1]+bounds[4])
    Zmax_crop = sitk_im.GetDepth() - (bounds[2] + bounds[5])

    cropped_img = sitk.Crop(sitk_im, [Xmin_crop, Ymin_crop, Zmin_crop], [Xmax_crop, Ymax_crop, Zmax_crop])
    cropped_mask = sitk.Crop(sitk_mask, [Xmin_crop, Ymin_crop, Zmin_crop], [Xmax_crop, Ymax_crop, Zmax_crop])

    return(cropped_img, cropped_mask)





