import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras

"""https://github.com/revsic/tf-neural-process/blob/master/neural_process/"""
def dense_sequential(output_sizes, activation=tf.nn.relu):
    model = tfk.Sequential()
    for size in output_sizes[:-1]:
        model.add(tfk.layers.Dense(size, activation=activation))
    
    model.add(tfk.layers.Dense(output_sizes[-1]))
    return model

# commented out because we should just have one encoder type
# class Encoder(tfk.layers.Layer):
#     def __init__(self, output_sizes, attention=None, keepdims=False, name='Encoder'):
#         super(Encoder, self).__init__(name=name, dynamic=True)
#         self.model = dense_sequential(output_sizes)
#         self.attention = attention
#         self.keepdims = keepdims

#     def call(self, rep, key=None, query=None):
#         if isinstance(rep, (tuple, list)):
#             rep = tf.concat(rep, axis=-1)

#         hidden = self.model(rep)
#         if self.attention is not None:
#             hidden = self.attention(query=query, key=key, value=hidden)
    
#         if not self.keepdims:
#             hidden = tf.reduce_mean(hidden, axis=1)

#         return hidden


class Decoder(tfk.layers.Layer):
    def __init__(self, output_sizes, name='Decoder'):
        super(Decoder, self).__init__(name=name, dynamic=True)
        self.model = dense_sequential(output_sizes)
    
    def call(self, context, tx):
        input_tensor = tf.concat([context, tx], axis=-1)
        return self.model(input_tensor)


class LatentEncoder(tfk.layers.Layer):
    def __init__(self, output_sizes, name='LatentVariable'):
        super(LatentEncoder, self).__init__(name=name, dynamic=True)
        self.model = dense_sequential(output_sizes)

    def call(self, rep):
        hidden = self.model(rep)
        # hidden = tf.reduce_mean(hidden, axis=1)
        # changed the hidden layer to be only the representations instead of learning the mu and sigma directly
        # mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half

        # sigma = tf.exp(log_sigma)

        mu = tf.reduce_mean(hidden, axis=1)
        sigma = tf.reduce_std(hidden, axis=1)
        # don't need the entire distribution here, because we reparametrize with a normal distribution
        # dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return mu, sigma


class NeuralProcess(tfk.Model):
    def __init__(self,
                 z_output_sizes,
                 enc_output_sizes,
                 dec_output_sizes, name='NeuralProcess'):
        super(NeuralProcess, self).__init__(name=name)

        self.z_encoder_latent = LatentEncoder(z_output_sizes)
        # self.encoder = Encoder(enc_output_sizes)
        self.decoder = Decoder(dec_output_sizes)#[:-1])

        # size = dec_output_sizes[-1]
        # self.dense_mu = tfk.layers.Dense(size)
        # self.dense_sigma = tfk.layers.Dense(size)
    
    @tf.function
    def call(self, x, eps):
        context, query = x

        z_mu, z_sigma = self.z_encoder_latent(context)

        # if we directly sample in the forward pass from the distribution, how will this be backpropogated?
        # latent = z_dist.sample()

        # dist = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(z_mu.shape), scale_diag=tf.ones(z_sigma.shape))
        # eps = dist.sample()
        latent = eps * z_sigma + z_mu

        latent = tf.tile(tf.expand_dims(latent, 1),
                          [1, tf.shape(query)[1], 1])

        rep = self.decoder(latent, query)
        t_mu, t_log_sigma = tf.split(rep, num_or_size_splits=2, axis=-1) # split the output in half
        # mu = self.dense_mu(rep)
        # log_sigma = self.dense_sigma(rep)
        
        t_sigma = tf.exp(t_log_sigma)

        return tf.concat([t_mu, t_sigma], axis=-1) #(dist, mu, sigma) # tf.concat([mu, sigma], axis=-1) #dist, mu, sigma
