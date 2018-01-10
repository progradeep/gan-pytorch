import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions

def sample_mog(batch_size, n_mixture=8, std=0.01, radius=2.0):
    thetas = np.linspace(0, 2*np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    sess = tf.Session()
    return sess.run(data.sample(batch_size))

if __name__ == '__main__':
    data = sample_mog(512)
    print(data)
