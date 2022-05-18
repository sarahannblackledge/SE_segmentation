from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Specify image and mask directories
im_dir = '/Users/sblackledge/Documents/SE_segmentation/Test/images'
#mask_dir = '/Users/sblackledge/Documents/SE_segmentation/Test/labels'

#Set seed to transform mask and image by same augmentation parameter
seed = 1

# ImageDataGenerator rotation
datagen = ImageDataGenerator(rotation_range=3, fill_mode='nearest', rescale=1./255)

# iterator
aug_iter = datagen.flow_from_directory(im_dir, class_mode=None, seed=seed, target_size=(256, 256), color_mode='grayscale')
#aug_iter_mask = datagen.flow_from_directory(mask_dir, class_mode=None, seed=seed)

# generate samples and plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
#fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))

# generate batch of images
for i in range(3):
    # convert to unsigned integers
    image = next(aug_iter)[0]
    #image2 = next(aug_iter_mask)[0].astype('uint8')


    # plot image
    ax[i].imshow(image, cmap='gray')
    #ax2[i].imshow(image2)

