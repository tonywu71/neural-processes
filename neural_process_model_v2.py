import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras

"""https://github.com/revsic/tf-neural-process/blob/master/neural_process/"""

def dense_sequential(output_sizes, activation=tf.nn.relu):
    """Convert number of hidden units to sequential dense layers
    Args:
        output_sizes: List[int], number of hidden units
        activation: Callable[[tf.Tensor], tf.Tensor], activation function, default ReLU
    Returns:
        tfk.Model, sequential model consists of dense layers
    """
    model = tfk.Sequential()
    for size in output_sizes[:-1]:
        model.add(tfk.layers.Dense(size, activation=activation))
    
    model.add(tfk.layers.Dense(output_sizes[-1]))
    return model

class Encoder(tfk.layers.Layer):
    """Context encoder
    Attributes:
        model: Callable[[tf.Tensor], tf.Tensor], dense sequential encoder
        attention: Callable[[tf.Tensor], tf.Tensor], attention method, default None
            if None, attention is not applied
        keepdims: bool, if false, reduce axis 1 with mean method
    """
    def __init__(self, output_sizes, attention=None, keepdims=False, name='Encoder'):
        super(Encoder, self).__init__(name=name, dynamic=True)
        self.model = dense_sequential(output_sizes)
        self.attention = attention
        self.keepdims = keepdims

    def call(self, rep, key=None, query=None):
        """Encode given context
        Args:
            rep: tf.Tensor or tuple, list of tf.Tensor, representation
                if rep consists of multiple tensor, concatenate it to decode
            key: tf.Tensor, key for attention method, default None
            query: tf.Tensor, query for attention method, default None
        Returns:
            tf.Tensor, encoded context
        """
        if isinstance(rep, (tuple, list)):
            rep = tf.concat(rep, axis=-1)

        hidden = self.model(rep)
        if self.attention is not None:
            hidden = self.attention(query=query, key=key, value=hidden)
    
        if not self.keepdims:
            hidden = tf.reduce_mean(hidden, axis=1)

        return hidden


class Decoder(tfk.layers.Layer):
    """Context decoder
    Attributes:
        model: Callable[[tf.Tensor], tf.Tensor], dense sequential decoder
    """
    def __init__(self, output_sizes, name='Decoder'):
        super(Decoder, self).__init__(name=name, dynamic=True)
        self.model = dense_sequential(output_sizes)
    
    def call(self, context, tx):
        """Decode tensor
        Args:
            context: tf.Tensor, encoded context
            tx: tf.Tesnor, query
        
        Returns:
            tf.Tensor, decoded value
        """
        input_tensor = tf.concat([context, tx], axis=-1)
        return self.model(input_tensor)


class GaussianProb(tfk.layers.Layer):
    """Convert input tensor to gaussian distribution representation
    Attributs:
        dense_mu: Callable[[tf.Tensor], tf.Tensor], dense layer for mean
        dense_sigma: Callable[[tf.Tensor], tf.Tensor], dense layer for sigma
        multivariate: bool, if true, return multivariate gaussian distribution with diagonal covariance
        proj: Callable[[tf.Tensor], tf.Tensor], projection layer for input tensor, default None
            if None, projection is not applied
    """
    def __init__(self, size, multivariate=False, proj=None, name='GP'):
        super(GaussianProb, self).__init__(name=name, dynamic=True)
        self.dense_mu = tfk.layers.Dense(size)
        self.dense_sigma = tfk.layers.Dense(size)
        self.multivariate = multivariate

        self.proj = proj
        if proj is not None:
            self.proj = tfk.layers.Dense(proj)
    
    def call(self, input_tensor):
        if self.proj is not None:
            input_tensor = self.proj(input_tensor)
        
        mu = self.dense_mu(input_tensor)
        log_sigma = self.dense_sigma(input_tensor)
        
        sigma = tf.exp(log_sigma)
        if self.multivariate:
            dist = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        else:
            dist = tfp.distributions.Normal(mu, sigma)
        
        return dist, mu, sigma 

class NeuralProcess(tfk.Model):
    """Neural Process
    Attributes:
        z_encoder: Encoder, encoder for latent representation
        z_prob: GaussianProb, latent representation to probability distribution
        encoder: Encoder, context encoder
        decoder: Decoder, decoder for context and latent variable
        normal_dist: GaussianProb, converter for decoded context to probability distribution
    """
    def __init__(self,
                 z_output_sizes,
                 enc_output_sizes,
                 dec_output_sizes, name='Neural Process'):
        """Initializer
        Args:
            z_output_sizes: List[int], number of hidden units for latent representation encoder
            enc_output_sizes: List[int], number of hidden units for context encoder
            dec_output_sizes: List[int], number of hidden units for decoder
        """
        super(NeuralProcess, self).__init__(name=name)

        self.z_encoder = Encoder(z_output_sizes[:-1])
        self.z_prob = GaussianProb(z_output_sizes[-1],
                                   proj=np.mean(z_output_sizes[-2:]))

        self.encoder = Encoder(enc_output_sizes)
        self.decoder = Decoder(dec_output_sizes[:-1])
        #self.normal_dist = GaussianProb(dec_output_sizes[-1], multivariate=True)
        size = dec_output_sizes[-1]
        self.dense_mu = tfk.layers.Dense(size)
        self.dense_sigma = tfk.layers.Dense(size)
    
    @tf.function
    def call(self, x):
        context, query = x
        #context_x, context_y, target_x = inputs # pixel coords, pixel values, all image coords, target image
        #context = context_y # context pixel values
        #context = tf.concat([context_x, context_y], axis=-1)
        #query = target_x # the pixel coords we want to predict - all image coords
        z_context = self.z_encoder(context)
        z_dist, _, _ = self.z_prob(z_context)

        latent = z_dist.sample()
        context = self.encoder(context)

        context = tf.concat([latent, context], axis=-1)
        context = tf.tile(tf.expand_dims(context, 1),
                          [1, tf.shape(query)[1], 1])

        rep = self.decoder(context, query)
        # dist, mu, sigma = self.normal_dist(rep)
        mu = self.dense_mu(rep)
        log_sigma = self.dense_sigma(rep)
        
        sigma = tf.exp(log_sigma)

        return tf.concat([mu, sigma], axis=-1)#(dist, mu, sigma) # tf.concat([mu, sigma], axis=-1) #dist, mu, sigma
    
    # def loss(self, context, query, target):
    #     dist, _, _ = self(context, query)
    #     log_prob = dist.log_prob(target)
    #     log_prob = tf.reduce_sum(log_prob)

    #     prior, _, _ = self.z_prob(self.z_encoder(context))
    #     posterior, _, _ = self.z_prob(self.z_encoder([query, target]))

    #     kl = tfp.distributions.kl_divergence(prior, posterior)
    #     kl = tf.reduce_sum(kl)

    #     # maximize variational lower bound
    #     loss = -log_prob + kl
    #     return loss
