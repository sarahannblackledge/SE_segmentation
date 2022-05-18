import SimpleITK as sitk
import numpy as np

#inputs:
#1. original_CT: variable name - sitk image object of image to be resampled
#2. desired_dimensions: list of desired output dimensions (i.e. desired_dimensions = [256, 256, 130]

def resample_patient(original_CT, desired_dimensions, is_label=False):
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    sz = original_CT.GetSize()
    # reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()]
    reference_size = desired_dimensions
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    # centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)
    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    if is_label:
        outimage = sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkNearestNeighbor, 0)
    else:
        outimage = sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, -1024)

    return outimage