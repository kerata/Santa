from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces


class Santa(Optimizer):
    """
        Euler implementation of Santa optimization algorithm,
        Adapted from theano implementation of authors
    """

    def __init__(self, lr, exploration, rho=0.95,
                 anne_rate=0.5, epsilon=1e-8, **kwargs):
        # default value for clipping is 'clip_norm = 5'
        if 'clipnorm' not in kwargs and 'clipvalue' not in kwargs:
            kwargs['clipnorm'] = 5

        super(Santa, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='float32', name='iterations')
            self.lr = K.variable(lr, name='lr')

            self.exploration = K.variable(exploration, name='exploration', dtype='float32')

            # anne_rate -> η
            self.anne_rate = K.variable(anne_rate, name='anne_rate')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        # rho -> σ
        self.rho = rho

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        # grads -> f_tilda
        grads = self.get_gradients(loss, params)

        ###
        ### Removed part seems to be related theano specifically
        ###

        # i = theano.shared(numpy_floatX(0.))
        i_t = self.iterations + 1

        # Exploration condition (CHANGED!)
        should_explore = K.cast(K.less(i_t, self.exploration), K.floatx())

        # Inverse temperature β
        b_t = K.pow(i_t, self.anne_rate)

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        als = [K.ones(K.int_shape(p), dtype=K.dtype(p)) * .5 for p in params]

        for p, g, m, v, a in zip(params, grads, ms, vs, als):
            # m = K.variable(p.get_value() * 0.)
            # v = K.variable(p.get_value() * 0.)
            # alpha = K.variable(K.ones(p.get_value().shape) * .5)

            # In Exploitation value is not updated, (1 / 2) not found?
            a_t = a + should_explore * (K.pow(m, 2) - self.lr / b_t)

            # (1 / N^2) not found? (probably normalization factor...)
            v_t = self.rho * v + (1. - self.rho) * K.pow(g, 2)

            # pcder -> 1 / g_t
            pcder = K.sqrt(K.sqrt(v_t) + self.epsilon)

            # eps -> ζ : standard normal random vector
            # eps = K.random_normal(p.get_value().shape, mean=0.0, stddev=1.0)
            eps = K.random_normal(K.int_shape(p), mean=0.0, stddev=1.0)

            # m_t -> u_t
            # (1 - g_t-1 / g_t) term omitted for complexity
            # g_t-1 term is replaced with a (not constant?!) self.nframes
            # 1 instead of (v_t / self.nframes) since it can be constant
            m_t = (1. - a_t) * m - self.lr * g / pcder + should_explore * \
                K.sqrt((2 * self.lr / b_t) * v_t) * eps

            # p_t -> θ_t
            p_t = p + m_t / pcder

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(p, p_t))
            # self.updates.append(K.update(a, a_t))
        # Adam implementation did not updated iteration...
        # self.updates.append(K.update(i, i_t))

        return self.updates

    def get_config(self):
        # lr, exploration, rho=0.95,
        # anne_rate=0.5, epsilon=1e-8
        config = {
            'lr': float(K.get_value(self.lr)),
            'exploration': int(K.get_value(self.exploration)),
            'rho': self.rho,
            'anne_rate': float(K.get_value(self.anne_rate)),
            'epsilon': self.epsilon
        }
        # clip_norm=5
        base_config = super(Santa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

