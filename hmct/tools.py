import tensorflow as tf
import tensorflow_probability as tfp

class AHMC():
    def __init__(self, m, s, L, logp, ndim):
        self.m = m
        self.s = s
        self.L = L
        self.logp = logp
        self.ndim = ndim
        self.input_shape = (self.m, 2, self.ndim)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(1, activation="sigmoid"),
            tf.keras.layers.Lambda(lambda x: x/self.s)
        ])
    
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            epsilon_c = tf.squeeze(self.model(tf.expand_dims(self.data, 0)))
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.logp,
                num_leapfrog_steps=self.L,
                step_size=epsilon_c)
            
            results = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=0,
                current_state=tf.repeat([self.data[-1][0]], self.m, axis=0),
                kernel=hmc, trace_fn=lambda _, ker: (ker.proposed_results.final_momentum, ker.log_accept_ratio, ker.proposed_state)) 
            
            alpha = tf.math.minimum(tf.exp(results.trace[1]), 1.) # (1,self.m) (self.m, 2) (1,self.m, 2)
            sjd = tf.math.reduce_sum((tf.repeat([self.data[-1][0]], self.m, axis=0)-results.all_states[0])**2, axis=-1)
            esjd = tf.math.reduce_mean(alpha[0]*sjd)
            loss = 1/esjd

        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return results, epsilon_c
    
    def warm_up(self, epochs):
        optimizer = tf.keras.optimizers.Adam()
        self.saved_epsilon = tf.constant([])
        for e in range(epochs):
            out, epsilon = self.train_step(optimizer)
            index = tf.random.uniform([], maxval=self.m, dtype=tf.int32)
            self.data = tf.concat([self.data, [[out.all_states[0][index], out.trace[0][0][0][index]]]], axis=0)
            self.data = self.data[-self.m:]
            self.saved_epsilon = tf.concat([self.saved_epsilon, [epsilon]], axis=0)
        self.epsilon = tf.math.reduce_mean(self.saved_epsilon)
        
    def hmc(self, n, epsilon=None, initial_pos=None):
        if epsilon is None:
            epsilon = self.epsilon
        if initial_pos is None:
            initial_pos = [self.data[-1][0]]
        results = self.sample_chain(epsilon, n, initial_pos, None)
        
        return results[:, 0, :]
        
    def collect_training_data(self, epsilon, initial_pos):
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.logp,
            num_leapfrog_steps=self.L,
            step_size=epsilon)
        
        results = tfp.mcmc.sample_chain(
            num_results=self.m,
            num_burnin_steps=0,
            current_state=tf.cast(initial_pos, tf.float32),
            kernel=hmc, trace_fn=lambda _, ker: ker.proposed_results.final_momentum)
        
        self.data = tf.reshape(tf.stack([results.all_states, results.trace[0]], axis=-1), self.input_shape)

class HMC():
    def __init__(self, e, L, logp):
        self.e = e
        self.L = L
        self.logp = logp
        self.kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.logp,
            num_leapfrog_steps=self.L,
            step_size=self.e)
    
    def hmc(self, n, burn_in, current_state):
        return tfp.mcmc.sample_chain(
            num_results=n,
            num_burnin_steps=burn_in,
            current_state=current_state,
            kernel=self.kernel)

class NUTS():
    def __init__(self, e, logp):
        self.e = e
        self.logp = logp
        self.kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.logp,
                step_size=self.e)
    
    def hmc(self, n, burn_in, current_state):
        return tfp.mcmc.sample_chain(
            num_results=n,
            num_burnin_steps=burn_in,
            current_state=current_state,
            kernel=self.kernel)