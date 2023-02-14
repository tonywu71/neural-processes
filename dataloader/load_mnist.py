from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds







def load_mnist(batch_size: int=32, num_context_points=None, uniform_sampling = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Loads the mnist dataset

    Args:
        batch_size (int, optional): model batch size. Defaults to 32.
        num_context_points (int/None, optional): number of context points to use. Defaults to None randomly sampling between 10 and 100.
        uniform_sampling (bool): Whether to uniformly sample the context points (True) or to order samples (False)

    Returns:
        (train, test) datasets
    """
    mnist = tfds.load('mnist')  # Note: By default, autocaching is enabled on MNIST
    train_ds, test_ds = mnist['train'], mnist['test']


    def encode(element):
        # element should be already batched
        img = tf.cast(element['image'], tf.float32) / 255. # normalise pixels within [0,1] range
        batch_size = tf.shape(img)[0] 
        
        if num_context_points is None:
            # Number observations of the target image randomly chosen between [10,100]
            num_context = tf.random.uniform(
                shape=[], minval=10, maxval=100, dtype=tf.int32)
        else:
            num_context = num_context_points

        # For each of our observations, sample x,y coordinates in range [0,27]
            context_x = tf.random.uniform(
                shape=(batch_size, num_context, 2),
                minval=0, maxval=27, dtype=tf.int32)
            
        if not uniform_sampling:
            context_x = tf.sort(context_x, axis=1)
        
        # Sample observation coordinates from target image
        context_y = tf.gather_nd(img, context_x, batch_dims=1)

        # Normalise the observation coordinates to the range [0,27] to be model size agnostic
        context_x = tf.cast(context_x, tf.float32)  /27.

        # define the grid of x,y coordinates for every pixel
        cols, rows = tf.meshgrid(tf.range(28.), tf.transpose(tf.range(28.)))

        # combine the x,y coordinate arrays into a single array
        grid = tf.stack([rows, cols], axis=-1)  # (28, 28, 2)

        # copy observation coordinates across the entire batch
        batch_grid = tf.tile(
            tf.expand_dims(grid, axis=0),
            [batch_size, 1, 1, 1])  # (batch_size, 28, 28, 2)
        
        # Normalise the observation coordinates to the range [0,27] to be model size agnostic
        target_x = tf.reshape(
            batch_grid, (batch_size, 28 * 28, 2)) / 27.  # normalize
        
        # reshape the target image to have shape: batch size, input dim, 1 
        target_y = tf.reshape(img, (batch_size, 28 * 28, 1))
        return (context_x, context_y, target_x), target_y
    
    train_ds = train_ds.batch(batch_size).map(encode).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).map(encode).prefetch(tf.data.experimental.AUTOTUNE)  # TODO check if prefetch needed
    
    return train_ds, test_ds