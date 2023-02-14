from PIL import Image
import numpy as np
import random
import os
import tensorflow as tf
from tqdm import tqdm

def dataloader2(data_path):
	try:
		# Attempt to read cached data
		data = np.load(data_path + "images.npz")
		images = data['images']
	except FileNotFoundError:
		image_path = data_path + "img_align_celeba/img_align_celeba/"
		image_names = os.listdir(image_path)
		images = np.zeros((len(image_names),32,32, 3), dtype=np.uint8)

		for i, image_name in tqdm(enumerate(random.sample(image_names, len(image_names))), desc="images"):
			image = Image.open(image_path+image_name)
			image = image.resize((32, 32), Image.BICUBIC)
			images[i,:,:] = np.asarray(image)
			

		np.savez(data_path + "images.npz", images=images)
	return images




def load_celeb(batch_size: int=32, num_context_points=None, uniform_sampling = True):
    """Loads the celeb_a dataset

    Args:
        batch_size (int, optional): model batch size. Defaults to 32.
        num_context_points (int/None, optional): number of context points to use. Defaults to None randomly sampling between 10 and 100.
        uniform_sampling (bool): Whether to uniformly sample the context points (True) or to order samples (False)

    Returns:
        (train, test) datasets
    """
    celeb_a = dataloader2(".data/")

    train_num = int(celeb_a.shape[0] * 0.8)

    train_ds = celeb_a[0:train_num,:,:,:]
    test_ds = celeb_a[train_num:,:,:,:]
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)


    def encode(img):
        """samples context points from a batch of colour images"""
        # element should be already batched
        img_size = img.shape[1]
        img = tf.cast(img, tf.float32) / 255. # normalise pixels within [0,1] range
        batch_size = tf.shape(img)[0]

        if num_context_points is None:
            # Number observations of the target image randomly chosen between [10,100]
            num_context = tf.random.uniform(
                shape=[], minval=10, maxval=100, dtype=tf.int32)
        else:
            num_context = num_context_points

        # For each of our observations, sample x,y coordinates in range [0,32]
        context_x = tf.random.uniform(
            shape=(batch_size, num_context, 2),
            minval=0, maxval=(img_size-1), dtype=tf.int32)

        if not uniform_sampling:
            context_x = tf.sort(context_x, axis=1)

        # Sample observation coordinates from target image
        context_y = tf.gather_nd(img, context_x, batch_dims=1) 
        
        # Normalise the observation coordinates to the range [0,32] to be model size agnostic
        context_x = tf.cast(context_x, tf.float32)  / float(img_size-1.0) 
        
        # define the grid of x,y coordinates for every pixel
        cols, rows, chan = tf.meshgrid(tf.range(float(img_size)), tf.transpose(tf.range(float(img_size))), tf.range(float(3)))
        
        # combine the x,y coordinate arrays into a single array
        grid = tf.stack([rows, cols, chan], axis=-1)  # (32, 32,3, 3)
        
        # copy observation coordinates across the entire batch
        batch_grid = tf.tile(
            tf.expand_dims(grid, axis=0),
            [batch_size, 1, 1, 1, 1])  # (batch_size, 32, 32, 3, 3)
        
        # Normalise the observation coordinates to the range [0,27] to be model size agnostic
        target_x = tf.reshape(batch_grid, (batch_size, img_size * img_size*3, 3))
        target_x = target_x  / float(img_size-1.0)  # normalize

        # reshape the target image to have shape: batch size, input dim, 1 
        target_y = tf.reshape(img, (batch_size, img_size * img_size * 3, 1))
        return (context_x, context_y, target_x), target_y

    train_ds = train_ds.batch(batch_size).map(encode)
    test_ds = test_ds.batch(1).map(encode)

    return train_ds, test_ds