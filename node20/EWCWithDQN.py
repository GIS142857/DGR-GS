import random

import numpy as np
import tensorflow as tf
tf.disable_v2_behavior()
import os


checkpoint_base_dir = 'checkpoints_tutorial16/'

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reward_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None


def update_paths():
    """
    Update the path-names for the checkpoint-dir and log-files.

    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path

    # Add the environment-name to the checkpoint-dir.
    checkpoint_dir = os.path.join(checkpoint_base_dir, "EWCLearning")

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")


class ENV:
    def __init__(self, state_dim, graph):
        self.state_dim = state_dim ### D, k, h
        self.graph = graph

    def reset(self, D, k):
        state = []
        state.append(D)
        state.append(k)
        state.append(0)
        return state

    def random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        print('x',x)
        cumulative_probability = 0.0
        print('some_list', some_list)
        print('probabilities', probabilities)
        if len(some_list) == 1:
            return some_list[0]
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                print('item', item)
                return item

    def step(self, state, node, action):  #### return next_state, reward, info, node_id
        key = str(action) + ',' + str(state[1]) + ',' + str(state[2]+1)
        real_d = self.graph.node_list[node].nb_delay[key]
        real_prob = self.graph.node_list[node].nb_delay_pro[key]
        print('key', key)
        print('real_d', self.graph.node_list[node].nb_delay)
        print('real_prob', self.graph.node_list[node].nb_delay_pro)
        reward = self.random_pick(real_d, real_prob)
        print('reward', reward)
        next_state = []
        d = state[0]-reward
        next_state.append(d)
        next_state.append(state[1])
        next_state.append(state[2]+1)
        info = False
        if next_state == self.graph.node_num:
            info = True
        return next_state, reward, info

class LinearControlSignal:
    """
    A control signal that changes linearly over time.
    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.

    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):

        """
        Create a new object.
        :param start_value:
            Start-value for the control signal.
        :param end_value:
            End-value for the control signal.
        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.
        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value

class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.

    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.
    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.
    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e4,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """

        :param num_actions:
            Number of possible actions in the game-environment.
        :param epsilon_testing:
            Epsilon-value when testing.
        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.

        :param start_value:
            Starting value for linearly decreasing epsilon.
        :param end_value:
            Ending value for linearly decreasing epsilon.
        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.

        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.

        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.
        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.
        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon


class ReplayMemory:

    def __init__(self, size, num_actions, state_dim, discount_factor=1):
        """

        :param size:
            Capacity of the replay-memory. This is the number of states.
        :param num_actions:
            Number of possible actions in the game-environment.
        :param discount_factor:
            Discount-factor used for updating Q-values.
        """
        # Estimation errors for the Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

        self.reset_memory_size(num_actions, state_dim)

    def reset_memory_size(self, num_actions, state_dim):
        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[self.size, state_dim], dtype=np.float)

        self.next_state = np.zeros(shape=[self.size, state_dim], dtype=np.float)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[self.size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[self.size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=self.size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=self.size, dtype=np.float)

        # Whether the life had ended in each state of the game-environment.
        self.end_life = np.zeros(shape=self.size, dtype=np.bool)

        self.estimation_errors = np.zeros(shape=self.size, dtype=np.float)

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, next_state, end_life):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state
        :param q_values:
            The estimated Q-values for the state.
        :param action:
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        :param end_life:
            Boolean whether the agent has lost a life in this state.
        """

        # Index into the arrays for convenience.
        k = self.num_used % self.size

        # Increase the number of used elements in the replay-memory.
        self.num_used += 1

        # Store all the values in the replay-memory.
        self.states[k] = state
        self.q_values[k] = q_values
        self.actions[k] = action
        self.end_life[k] = end_life

        # Note that the reward is limited. This is done to stabilize
        # the training of the Neural Network.
        # self.rewards[k] = np.clip(reward, -1.0, 1.0)
        self.rewards[k] = reward
        self.next_state[k] = next_state


    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used - 1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                # valid_q = self.q_values[k + 1][valid_actions]
                # Reference [1] equation in algorithm 1
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value


    def prepare_sampling_prob(self, batch_size=32):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.
        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.

        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err < self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.
        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))
        # idx = np.random.choice(np.arange(self.num_used), 128, replace=False)

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def all_batches(self, batch_size=32):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.

        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end


    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))


class NeuralNetwork:
    """
    Creates a Neural Network for Reinforcement Learning (Q-Learning).
    Functions are provided for estimating Q-values from states of the
    game-environment, and for optimizing the Neural Network so it becomes
    better at estimating the Q-values.
    """

    def __init__(self, num_actions, state_dim, replay_memory, agent_id, session, load_checkpoint=False):
        """
        :param num_actions:
            Number of discrete actions for the game-environment.
        :param replay_memory:
            Object-instance of the ReplayMemory-class.
        :param use_pretty_tensor:
            Boolean whether to use PrettyTensor (True) which must then be
            installed, or use the tf.layers API (False) which is already
            built into TensorFlow.
        """

        self.use_pretty_tensor = False

        # Set valid action indices
        # self.action_indices_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        # self.action_indices = action_indices

        # Replay-memory used for sampling random batches.
        self.replay_memory = replay_memory

        # Path for saving/restoring checkpoints.
        path_name = str(agent_id) + 'checkpoint'
        print('path_name', path_name)
        print('checkpoint_dir', checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, path_name)
        print('path', self.checkpoint_path)
        #self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")  #


        # Placeholder variable for inputting the learning-rate to the optimizer.
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Placeholder variable for inputting the target Q-values
        # that we want the Neural Network to be able to estimate.
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, None],
                                           name='q_values_new')

        # This is a hack that allows us to save/load the counter for
        # the number of states processed in the game-environment.
        # We will keep it as a variable in the TensorFlow-graph
        # even though it will not actually be used by TensorFlow.
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')

        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')

        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)

        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        # Tensor for the fisher matrix
        self.fisher_tensor = tf.Variable(initial_value=0,
                                         trainable=False, dtype=tf.float32,
                                         name='fisher')

        self.state_input = tf.placeholder("float", [None, state_dim])

        # Init for weights
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)
        self.bias_init = tf.constant_initializer(0)

        activation = tf.nn.relu

        # First fully-connected (aka. dense) layer.
        net = tf.layers.dense(inputs=self.state_input, name='layer1', units=128,
                              kernel_initializer=init, activation=activation)

        # Second fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer2', units=64,
                              kernel_initializer=init, activation=activation)

        # Final fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer3', units=num_actions,
                              kernel_initializer=init, activation=None)

        self.q_values = net


        # L2-loss
        squared_error = tf.square(self.q_values - self.q_values_new)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        self.loss = tf.reduce_mean(sum_squared_error)
        self.ewc_loss = self.loss

        # Optimizer used for minimizing the loss-function.
        # Note the learning-rate is a placeholder variable so we can
        # lower the learning-rate as optimization progresses.
        # NOTE: Not currently used with EWCLearning, Adam used instead
        # self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = self.rmsprop.minimize(self.loss)

        # Used for saving and loading checkpoints.
        self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        # self.session = tf.Session()
        self.session = session

        # Make dictionary to remember biases specific to each game
        # Reference [3] equation S34
        self.bias_history = {}

        layer_names = ["layer1", "layer2", "layer3"]
        self.var_list = []
        self.bias_list = []
        for layer in layer_names:
            self.var_list.append(self.get_weights_variable(layer))
            self.bias_list.append(self.get_bias_variable(layer))

        # ADAM Optimization
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.adam = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon)
        self.adam_train_step = self.adam.minimize(self.ewc_loss)

        # Make last_m, and last_v an array with same len as self.var_list
        self.last_m = []
        self.last_v = []
        self.ewc_loss_fisher_part = 0
        for v in self.var_list:
            self.last_m.append(np.zeros(np.shape(v)))
            self.last_v.append(np.zeros(np.shape(v)))
        self.last_t = 1

        self.fisher = []
        for v in range(len(self.var_list)):
            self.fisher.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        if load_checkpoint:
            self.load_checkpoint()
        else:
            self.session.run(tf.global_variables_initializer())

    def close(self):
        """Close the TensorFlow session."""
        self.session.close()

    def compute_fisher(self):
        for var in range(len(self.var_list)):
            # Update fisher information matrix for each variable
            v = self.adam.get_slot(self.var_list[var], "v")
            # Reference [2] equation 62, 63
            self.fisher[var] = v / (1 - self.beta2 ** self.last_t)

    def update_ewc_loss(self, lam):
        # lam is weighting for previous task(s) constraints
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.loss

        for v in range(len(self.var_list)):
            # Update ewc loss using the fisher matrix
            self.ewc_loss_fisher_part += (lam / 2) * tf.reduce_sum(
                tf.multiply(self.fisher[v], tf.square(self.var_list[v] - self.star_vars[v])))
            # Reference [3] equation 3
            self.ewc_loss += (lam / 2) * tf.reduce_sum(
                tf.multiply(self.fisher[v], tf.square(self.var_list[v] - self.star_vars[v])))
        # Update the optimizer
        self.adam_train_step = self.adam.minimize(self.ewc_loss)

    def save_optimal_weights(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval(session=self.session))

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                self.session.run(self.var_list[v].assign(self.star_vars[v]))

    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """

        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            self.fisher = self.session.run(self.fisher_tensor)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            self.session.run(tf.global_variables_initializer())

    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        print("Saved checkpoint.")

    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given states.
        The output of this function is an array of Q-value-arrays.
        There is a Q-value for each possible action in the game-environment.
        So the output is a 2-dim array with shape: [batch, num_actions]
        """
        # Create a feed-dict for inputting the states to the Neural Network.
        feed_dict = {self.state_input: states}

        # Use TensorFlow to calculate the estimated Q-values for these states.
        values = self.session.run(self.q_values, feed_dict=feed_dict)

        return values

    def optimize_adam(self, learning_rate, batch_size=32, max_epochs=10.0, min_epochs=1.0, loss_limit=0.015):

        self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        iterations_per_epoch = self.replay_memory.num_used / batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs) + self.last_t

        # Maximum number of iterations to perform.
        max_iterations = int(iterations_per_epoch * max_epochs)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        print("Optimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: {0:.1e}".format(learning_rate))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        for t in range(self.last_t, max_iterations + self.last_t):
            state_batch, q_values_batch = self.replay_memory.random_batch()

            feed_dict = {self.state_input: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}

            loss_val, _, loss_normal = self.session.run([self.ewc_loss, self.adam_train_step, self.loss],
                                                        feed_dict=feed_dict)
            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            if hasattr(self, "ewc_loss_fisher_part"):
                if self.ewc_loss_fisher_part != 0:
                    fisher_loss = self.session.run(self.ewc_loss_fisher_part, feed_dict=feed_dict)
                else:
                    fisher_loss = 0
            else:
                fisher_loss = 0
            # Print status.
            # pct_epoch = t / iterations_per_epoch
            # msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}, Fisher Loss: {4:.4f}, Q Loss: {5:.4f}"
            # msg = msg.format(t, pct_epoch, loss_val, loss_mean, fisher_loss, loss_normal)

            if t > min_iterations and loss_mean < loss_limit:
                break

            self.last_t += 1
        # New line
        #print()

        # fisher_v = tf.Variable(initial_value=self.fisher, dtype=tf.float32)
        # self.session.run(tf.assign(self.fisher_tensor, self.fisher, validate_shape=False))

    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.
        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """

        if self.use_pretty_tensor:
            # PrettyTensor uses this name for the weights in a conv-layer.
            variable_name = 'weights'
        else:
            # The tf.layers API uses this name for the weights in a conv-layer.
            variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def get_bias_variable(self, layer_name):
        variable_name = "bias"

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def save_biases(self, game):
        self.bias_history[game] = self.bias_list

        for bias in self.bias_list:
            shape = tf.shape(bias)
            self.session.run(tf.assign(bias, tf.zeros(shape, dtype=tf.float32)))

    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights

    def get_layer_tensor(self, layer_name):
        """
        Return the tensor for the output of a layer.
        Note that this does not return the actual values,
        but instead returns a reference to the tensor
        inside the TensorFlow graph. Use get_tensor_value()
        to get the actual contents of the tensor.
        """

        # The name of the last operation of a layer,
        # assuming it uses Relu as the activation-function.
        tensor_name = layer_name + "/Relu:0"

        # Get the tensor with this name.
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.state_input : [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.session.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_episodes_increase)

    def restore_bias(self, game):
        biases = self.bias_history[game]

        for i in range(len(biases)):
            self.session.run(tf.assign(self.bias_list[i], biases[i]))
