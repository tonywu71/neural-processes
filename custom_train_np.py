from collections import defaultdict
from typing import DefaultDict, Dict
from tqdm.auto import trange

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def train_np_model(model: tf.keras.Model,
                   ds_train: tf.data.Dataset,
                   ds_test: tf.data.Dataset,
                   optimizer: tf.keras.optimizers.Optimizer,
                   epochs:int,
                   verbose: bool=True) -> Dict[str, DefaultDict[int, float]]:
    """Custom training loop in Tensorflow for a Neural Process model."""
    
    train_loss_results = defaultdict(list)
    test_loss_results = defaultdict(list)
        
    # Initialize metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Start training
    for epoch in trange(epochs, desc='Training'):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        # Training step
        for (context_x, context_y, target_x), target_y in ds_train:
            with tf.GradientTape() as tape:
                # Compute the loss
                pred_y = model(context_x, context_y, target_x)
                
                mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

                log_prob = dist.log_prob(target_y)
                log_prob = tf.reduce_sum(log_prob)

                context = tf.concat([context_x, context_y], axis=-1)
                
                prior = model.z_encoder_latent(context) # TODO can we even call methods of the model in the loss?
                posterior = model.z_encoder_latent(tf.concat([target_x, target_y], axis=-1))

                kl = tfp.distributions.kl_divergence(prior, posterior)
                kl = tf.reduce_sum(kl)

                # maximize variational lower bound
                loss_value = -log_prob + kl
                
            # Compute the gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
            # Update the weights
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Update the metrics
            train_loss(loss_value)

        # Test step
        for (context_x, context_y, target_x), target_y in ds_test:
            pred_y = model(context_x, context_y, target_x)
                
            mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

            log_prob = dist.log_prob(target_y)
            log_prob = tf.reduce_sum(log_prob)

            context = tf.concat([context_x, context_y], axis=-1)
            
            prior = model.z_encoder_latent(context) # TODO can we even call methods of the model in the loss?
            posterior = model.z_encoder_latent(tf.concat([target_x, target_y], axis=-1))

            kl = tfp.distributions.kl_divergence(prior, posterior)
            kl = tf.reduce_sum(kl)

            # maximize variational lower bound
            loss_value = -log_prob + kl
            
            test_loss(loss_value)
            
        # End epoch
        train_loss_results[epoch].append(train_loss.result())
        test_loss_results[epoch].append(test_loss.result())
            

        if verbose:
            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            print(template.format(epoch+1,
                                  train_loss.result(),
                                  test_loss.result()))

    history = {"train": train_loss_results, "test": test_loss_results}
    
    return history
