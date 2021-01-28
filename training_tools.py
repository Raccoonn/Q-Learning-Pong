
import numpy as np
import matplotlib.pyplot as plt



def load_setup(filename):
    """
    Load in constants and hyper parameters for network and training scheme

    Format as follows:  - Basic information
                        - Changes to Environment (if any)
                        - save_dir
                        - p1_type
                        - p2_type
                        - mem_size
                        - action_space,  eg.  -30,0,30
                        - batch_size
                        - h1_dims
    """

    with open(filename) as f:
        raw = f.read().splitlines()

    info = raw[0]
    env_info = raw[1]
    save_dir = raw[2]
    p1_type = raw[3]
    p2_type = raw[4]
    mem_size = int(raw[5])
    action_Space = [int(n) for n in raw[6].split(',')]
    batch_size = int(raw[7])
    h1_dims = int(raw[8])

    return info, env_info, save_dir, p1_type, p2_type, mem_size, action_Space, batch_size, h1_dims          






class ReplayBuffer(object):
    """
    Memory buffer used to store states and sample during training
    """
    def __init__(self, max_size, input_shape, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.state__memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


    def store_transition(self, state, action, reward, state_, done):
        """
        Store values for current and next state
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state__memory[index] = state_
        
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        """
        Draw a sample from the store memory of states, actions, and rewards
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.state__memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal





def plot_progress(p1_tot, p2_tot,
                  p1_rwd_store, p2_rwd_store,
                  rallies, rally_store, episode, avg_rallies,
                  p1_avg, p2_avg,
                  save_dir
                  ):
    """
    Plot training progress to file
    """
    p1_rwd_store.append(p1_tot)
    p2_rwd_store.append(p2_tot)
    p1_avg.append(np.mean(p1_rwd_store))
    p2_avg.append(np.mean(p2_rwd_store))

    rally_store.append(rallies)
    avg_rallies.append(np.mean(rally_store))

    eps = list(range(episode)) 

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Rallies')
    line1, = ax1.plot(eps, avg_rallies, color='red', label='Avg Rallies')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Reward')
    line2, = ax2.plot(eps, p1_avg, color='blue',  label='P1 Avg')
    line3, = ax2.plot(eps, p2_avg, color='green', label='P2 Avg')

    plt.legend((line1, line2, line3),
               ('Avg Rallies', 'P1 Avg Reward', 'P2 Avg Reward'),
               loc='lower right')

    plt.savefig(save_dir + 'progress.png')
    plt.close()