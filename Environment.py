from unityagents import UnityEnvironment


class Environment:
    def __init__(self, environment_file):
        self.env = UnityEnvironment(file_name=environment_file)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)

        # examine the state space
        self.states = env_info.vector_observations
        self.state_size = self.states.shape[1]
        print(
            'There are {} agents. Each observes a state with length: {}'.format(
                self.states.shape[0],
                self.state_size)
        )
        print('The state for the first agent looks like:', self.states[0])

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        initial_state = env_info.vector_observations[0]

        return initial_state

    def step(self, action):
        env_info = self.env.step(action.cpu().numpy())[self.brain_name]
        state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]

        return state, reward, done
