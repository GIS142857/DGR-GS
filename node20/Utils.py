import random
import numpy as np
import operator
from collections import defaultdict

class ENV:
    def __init__(self, state_dim, graph):
        self.state_dim = state_dim ### D, k, h
        self.graph = graph

    def reset(self, source, des, D, k):
        state = []
        state.append(D)
        state.append(k)
        state.append(source + 1)
        state.append(des + 1)
        #state.append(1.0)
        return state

    def random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        l = len(some_list)
        #print('x',x)
        cumulative_probability = 0.0
        # print('some_list', some_list)
        # print('probabilities', probabilities)
        if len(some_list) == 1:
            return some_list[0]

        i = 0
        for item, item_probability in zip(some_list, probabilities):
            i += 1
            cumulative_probability += item_probability
            #print('cumulative_probability', cumulative_probability)
            if x < cumulative_probability:
                #print('item', item)
                return item
        return some_list[l-1]

    def get_queueing_time(self, real_d, k):
        print('real_d', real_d)
        real_d = real_d/1000.0
        if k == 1:
            # print('arrive', self.arrives[k-1],'real_d', real_d)
            # print('arrive', self.arrives[k], 'real_d', real_d)
            # print('(1 - self.arrives[k-1] * real_d)', (1 - self.arrives[k-1] * real_d))
            # print('222', (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d))
            if (1 - self.arrives[k-1] * real_d)<0.00000001 or (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d) < 0.00000000000001:
                #print('true')
                return 1000
            q = (self.arrives[k-1] * real_d * real_d)*(self.arrives[k] * real_d * real_d)/ (
                        (1 - self.arrives[k-1] * real_d) * (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d))
        else:
            # print('(1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d)', (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d))
            # print('22222', (1 - self.arrives[k-2] * real_d -self.arrives[k-1] * real_d-self.arrives[k] * real_d))
            if (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d)<0.0000001 or (1 - self.arrives[k-2] * real_d -self.arrives[k-1] * real_d-self.arrives[k] * real_d)<0.000000001:
                return 1000
            q = (self.arrives[k-2] * real_d * real_d )*(self.arrives[k-1] * real_d * real_d )*(self.arrives[k] * real_d * real_d )/ (
                         (1 - self.arrives[k-1] * real_d-self.arrives[k] * real_d)*(1 - self.arrives[k-2] * real_d -self.arrives[k-1] * real_d-self.arrives[k] * real_d))
        return q*1000

    def get_reward(self, path, h, flow): ### h is the hop of node
        l = len(path)
        node = path[l-2]
        action = path[l-1]
        p = self.graph[node][action]['per']
        send_num = 0
        delay = 0
        while send_num < 10:
            delay += self.graph.packet_len / (self.graph.trans[flow]) + self.graph.ave_channel_d
            rand = np.random.rand()
            if rand < p:
                break
            send_num += 1

        pdr_dot = 1
        for i in range(len(path) - 1):
            # print('i', i, 'node', path[i])
            pdr_dot *= (1 - self.graph.node_list[int(path[i])].pdr)

        # print('id', id, 'node', node)

        arrival_rate = self.graph.R[flow] * pdr_dot
        delay *= arrival_rate

        return delay


    def step(self, state, node, action, normal, training):  #### return next_state, reward, info, node_id
        #print('node', node, 'action', action)
        k = int(state[1] * normal[1] - 1)
        #h = int(state[2] * normal[2])
        key = str(action) + ',' + str(k) #+ ',' + str(h)
        real_d = self.graph.node_list[node].nb_delay[key]
        real_prob = self.graph.node_list[node].nb_delay_pro[key]
        reward = self.random_pick(real_d, real_prob)

        if training == False:
            reward = real_d[0]

        next_state = []
        d = state[0] * normal[0] - reward
        next_state.append(d)
        next_state.append(k+1.0)
        next_state.append(state[2] * normal[2])
        next_state.append(state[3] * normal[3])
        des = state[3] * normal[3] - 1
        #print('des', des)
        #next_state.append(h+1.0)
        info = 0
        if action == des:
            info = 1
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
        #print('num_iter', num_iterations)
        num_iterations = max(1, num_iterations)

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
                 num_iterations=1.5*1e3,
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

    def __init__(self, size, state_dim, discount_factor=1):
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

        self.reset_memory_size()


    def reset_memory_size(self):
        self.store = defaultdict(dict)
        self.num_used = 0
        self.ex_list = []
        self.index = defaultdict(dict)

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        print('num_used', self.num_used, 'size', self.size)
        return self.num_used / self.size

    # def reset(self):
    #     """Reset the replay-memory so it is empty."""
    #     self.num_used = 0

    def check(self, state, action, reward, next_state):
        #print('state', state, 'action', action, 'next_state', next_state, 'reward', reward)
        for i in range(self.num_used):
            # print('states', self.states[i])
            # print('actions', self.actions[i])
            # print('next_states', self.next_state[i])
            # print('rewards', self.rewards[i])
            if all(self.states[i] == state) and self.actions[i] == action and all(self.next_state[i] == next_state):
                #self.memory_updated = False
                #self.rewards[i] = reward
                return True, i
        #self.memory_updated = True
        return False, -1

    def add(self, key, data):  #data: state, action, action_index, reward, next_state, end_life
        # print('add_experience')
        # print('state', state, 'action', action, 'next_state', next_state, 'end_life', end_life)
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
        # Increase the number of used elements in the replay-memory.

        # bl, id = self.check(state, action, reward, next_state)
        # #print('bl', bl)
        # if bl:
        #     self.rewards[id] = reward
        #     return

        if key in self.store.keys():
            self.store[key].append(data)
            id = self.index[key]
            self.ex_list[id] = self.store[key]
        else:
            self.store[key] = []
            self.store[key].append(data)
            self.index[key] = self.num_used
            self.ex_list.append(self.store[key])
            if self.num_used < self.size:
                self.num_used += 1
        

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


    def prepare_sampling_prob(self, batch_size=1):
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

    def random_batch(self, batch_size):
        idx = random.sample((range(0, self.num_used)), batch_size)
        sample_list = []
        for i in range(batch_size):
            sample_list.append(self.ex_list[idx[i]])

        return sample_list

        # # Random index of states and Q-values in the replay-memory.
        # # These have LOW estimation errors for the Q-values.
        # idx_lo = np.random.choice(self.idx_err_lo,
        #                           size=self.num_samples_err_lo,
        #                           replace=False)
        #
        # # Random index of states and Q-values in the replay-memory.
        # # These have HIGH estimation errors for the Q-values.
        # idx_hi = np.random.choice(self.idx_err_hi,
        #                           size=self.num_samples_err_hi,
        #                           replace=False)
        #
        # # Combine the indices.
        # idx = np.concatenate((idx_lo, idx_hi))
        # # idx = np.random.choice(np.arange(self.num_used), 128, replace=False)
        #
        # # Get the batches of states and Q-values.
        # states_batch = self.states[idx]
        # q_values_batch = self.q_values[idx]
        #
        # return states_batch, q_values_batch

    def all_batches(self, batch_size=1):
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

        print('states', self.states)
        print('actions', self.actions)
        print('rewards', self.rewards)
        print('next_states', self.next_state)