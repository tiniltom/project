import tensorflow as tf
import numpy as np

def ResizeAndPad(images_, TARGET_SIZE_):
	
	maxLength = max(images_.shape[1], images_.shape[2])
	squaredImages =  tf.image.resize_image_with_crop_or_pad(images_, maxLength, maxLength)
	enlargedImages = tf.image.resize_image(squaredImages, [TARGET_SIZE_, TARGET_SIZE_])

	return enlargedImages
