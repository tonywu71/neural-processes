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

class Encoder(tfk.layers.Layer):
    def __init__(self, output_sizes, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.model = dense_sequential(output_sizes)
        self.hidden_output_shape = output_sizes[-1]

    @tf.function(reduce_retracing=True)
    def call(self, rep):
        batch_size, observation_points, context_dim = (tf.shape(rep)[0], tf.shape(rep)[1], tf.shape(rep)[2])
        hidden = tf.reshape(rep, shape=(batch_size * observation_points, context_dim))

        hidden = self.model(rep)

        outputs = tf.reshape(hidden, shape=(batch_size, observation_points, self.hidden_output_shape))

        outputs = tf.reduce_mean(outputs, axis=1)

        return outputs


class Decoder(tfk.layers.Layer):
    def __init__(self, output_sizes, name='Decoder'):
        super(Decoder, self).__init__(name=name)
        self.model = dense_sequential(output_sizes)
        self.output_size = output_sizes[-1]
    
    @tf.function(reduce_retracing=True)
    def call(self, context, tx):
        input_tensor = tf.concat((context, tx), axis=-1)

        batch_size, observation_points, input_dim = (tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], tf.shape(input_tensor)[2])
        input_tensor = tf.reshape(input_tensor, shape=(batch_size * observation_points, input_dim))

        outputs = self.model(input_tensor)

        outputs = tf.reshape(outputs, shape=(batch_size, observation_points, self.output_size))
        
        return outputs


class LatentEncoder(tfk.layers.Layer):
    def __init__(self, output_sizes, name='LatentVariable'):
        super(LatentEncoder, self).__init__(name=name)
        self.model = Encoder(output_sizes, name='LatentSequential')

    @tf.function(reduce_retracing=True)
    def call(self, rep):
        hidden = self.model(rep)

        mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half

        #sigma = tf.exp(log_sigma)
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        
        return dist


class NeuralProcessHybrid(tfk.Model):
    def __init__(self,
                 z_output_sizes,
                 enc_output_sizes,
                 dec_output_sizes, name='NeuralProcess'):
        super(NeuralProcessHybrid, self).__init__(name=name)

        self.z_encoder_latent = LatentEncoder(z_output_sizes)
        self.encoder = Encoder(enc_output_sizes)
        self.decoder = Decoder(dec_output_sizes)#[:-1])

        # size = dec_output_sizes[-1]
        # self.dense_mu = tfk.layers.Dense(size)
        # self.dense_sigma = tfk.layers.Dense(size)
    
    @tf.function(reduce_retracing=True)
    def call(self, x):
        # `context_x` shape (batch_size, observation_points, x_dim)
        # `context_y` shape (batch_size, observation_points, y_dim)
        context_x, context_y, query = x
        context = tf.concat((context_x, context_y), axis=-1)
        # `context` shape (batch_size, observation_points, x_dim + y_dim)
        
        z_dist = self.z_encoder_latent(context)
        latent = z_dist.sample()
        
        
        context = self.encoder(context)

        context = tf.concat((latent, context), axis=-1)
        target_points = tf.shape(query)[1]
        context = tf.tile(tf.expand_dims(context, 1),
                          (1, target_points, 1))

        rep = self.decoder(context, query)
        mu, log_sigma = tf.split(rep, num_or_size_splits=2, axis=-1) # split the output in half
        # mu = self.dense_mu(rep)
        # log_sigma = self.dense_sigma(rep)
        
        #sigma = tf.exp(log_sigma)
        # sigma = log_sigma
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        return tf.concat((mu, sigma), axis=-1)#(dist, mu, sigma) # tf.concat([mu, sigma], axis=-1) #dist, mu, sigma

    @tf.function(reduce_retracing=True)
    def compute_loss(self, x):
        (context_x, context_y, query), target_y = x
        context = tf.concat((context_x, context_y), axis=2)
        target_context = tf.concat((query, target_y), axis=2)


        pred_y = self(x[0])
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=2)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

        log_prob = dist.log_prob(target_y)
        log_prob = tf.reduce_sum(log_prob)

        prior = self.z_encoder_latent(context)
        posterior = self.z_encoder_latent(target_context)

        kl = tfp.distributions.kl_divergence(prior, posterior)
        kl = tf.reduce_sum(kl)

        # maximize variational lower bound
        loss = -log_prob + kl
        return loss

    @tf.function(reduce_retracing=True)
    def compute_loss_cnp(self, x):
        pred_y = self(x[0])
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=2)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(x[1])