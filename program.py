from torch import nn, optim, device, cuda, save
from Environment import Environment
from dqn_agent import Agent
from dqn import dqn, Params

device = device("cuda:0" if cuda.is_available() else "cpu")
# device="cpu"
print("***************************")
print("device", device)
print("***************************")
print()


def main():
    params = Params(n_episodes=50000,
                    max_t=1000,
                    eps_start=1.0,
                    eps_end=0.01,
                    eps_decay=0.995)

    environment = Environment('/Users/dali/workspace/RL/Reacher.app')
    # environment = Environment('/Users/dali/workspace/RL/Banana.app')
    agent = Agent(environment=environment, seed=0, device=device)
    dqn(agent, params)

    # env.close()


if __name__ == '__main__':
    main()
