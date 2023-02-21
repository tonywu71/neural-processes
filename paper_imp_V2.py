# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

# %%
from neural_process_model_v2 import NeuralProcess
from gp_curves import GPCurvesGenerator, plot_func

# %%
TRAINING_ITERATIONS = int(1e4) # 1e5
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(1e4)

# %%
dataset_train = GPCurvesGenerator(batch_size=64, max_size=MAX_CONTEXT_POINTS)
dataset_test = GPCurvesGenerator(batch_size=1, max_size=MAX_CONTEXT_POINTS, testing=True)
data_train = dataset_train.generate()



def gen(dataset):
    data_train = dataset.generate()
    for i in range(TRAINING_ITERATIONS):
        context = tf.concat((data_train[0][0], data_train[0][1]), axis=1)
        query = data_train[1]
        target = data_train[2]
        yield ((context, query), target)

train_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_train),
            output_types=((tf.float32, tf.float32), tf.float32)
        )

test_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_test),
            output_types=((tf.float32, tf.float32), tf.float32)
        )







# %%


z_output_sizes = [128, 128, 128, 128]
enc_output_sizes = [128, 128, 128, 128]
dec_output_sizes = [128, 128, 1]

model = NeuralProcess(z_output_sizes, enc_output_sizes, dec_output_sizes)



#opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

#%%
# def loss(target_y, pred_y):
#     dist, _, _ = pred_y#self(context, query)
#     log_prob = dist.log_prob(target)
#     log_prob = tf.reduce_sum(log_prob)

#     prior, _, _ = self.z_prob(self.z_encoder(context))
#     posterior, _, _ = self.z_prob(self.z_encoder([query, target]))

#     kl = tfp.distributions.kl_divergence(prior, posterior)
#     kl = tf.reduce_sum(kl)

#     # maximize variational lower bound
#     loss = -log_prob + kl
#     return loss


def loss(target_y, pred_y):
    # Get the distribution
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    #dist,mu, sigma = pred_y
    # mu = pred_y[1]
    # sigma = pred_y[2]
    
    dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    #dist = pred_y
    return -dist.log_prob(target_y)

model.compile(loss=loss, optimizer='adam')


# %%

model.fit(train_ds, epochs=1)

#%%
def plot():
    test_sample = next(gen(dataset_test))
    (context, query), target = test_sample
    pred_y = model(test_sample[0])
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    cx, cy = tf.split(context, num_or_size_splits=2, axis=1)
    plot_func(query, target, cx, cy, mu, sigma)
plot()
#%%
loss_graph = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_ITERATIONS):
        _, l = sess.run([opt, loss])
        loss_graph.append(l)

        if i % PLOT_AFTER == 0:
            context, query, target, loss_v, pred, var = sess.run([data_test.context,
                                                                  data_test.query,
                                                                  data_test.target,
                                                                  loss, mu, sigma])
            cx, cy = context
            print('Iteration: {}, loss: {}'.format(i, loss_v))

            plot_func(query, target, cx, cy, pred, var)

plt.plot(loss_graph)

# %%
z_output_sizes = [128, 128, 128, 128]
enc_output_sizes = [128, 128, 128, 128]
cross_output_sizes = [128, 128, 128, 128]
dec_output_sizes = [128, 128, 1]

self_attention = neural_process.Attention(attention_type='multihead', proj=[128, 128])
cross_attention = neural_process.Attention(attention_type='multihead', proj=[128, 128])

model = neural_process.AttentiveNP(z_output_sizes,
                                   enc_output_sizes,
                                   cross_output_sizes,
                                   dec_output_sizes,
                                   self_attention,
                                   cross_attention)

loss = model.loss(data_train.context,
                  data_train.query,
                  data_train.target)

_, mu, sigma = model(data_test.context, data_test.query)

opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

# %%
loss_graph = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_ITERATIONS):
        _, l = sess.run([opt, loss])
        loss_graph.append(l)

        if i % PLOT_AFTER == 0:
            context, query, target, loss_v, pred, var = sess.run([data_test.context,
                                                                  data_test.query,
                                                                  data_test.target,
                                                                  loss, mu, sigma])
            cx, cy = context
            print('Iteration: {}, loss: {}'.format(i, loss_v))

            plot_func(query, target, cx, cy, pred, var)

plt.plot(loss_graph)

# %%



