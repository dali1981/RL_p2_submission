[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

This project is an implementation of the DDPG model. 

### Distributed Training

No distributed training: the result is based on one agent. 

## The model

I use the DQN model as a base to my code. Instead of having a DNN approximating 
the value function and get the action with the highest value, the DNN models
 the policy and outputs the action. In our case a four-dimensional real array.
 
 The parameters of the models are as follow :
                    
        n_episodes=50000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995
        BUFFER_SIZE = int(1e5)  # replay buffer size

        BUFFER_SIZE = int(1e5)  # replay buffer size
        BATCH_SIZE = 1024  # minibatch size
        GAMMA = 0.99  # discount factor
        TAU = 1e-3  # for soft update of target parameters
        LR = 5e-4  # learning rate
        UPDATE_EVERY = 4  # how often to update the network
  
### Solving the Environment

The problem is not solved. My implementation leads to a score of zero as it 
trains. So I hope you can provide with guidance if there is a conceptual 
misunderstanding or a bug.

## Results

Reaults are displayed in output.txt

### Program

## program.py

This is where the main function resides. This is run with :
python program.py

## Environment.py

This is a wrapper to the environment that takes a step ahead with each action providing 
reward and the next state.

## dqn.py

Thh function dqn generates the environment path according to the actions provided by 
the model as the model trains.

## model.py

This is a classical DNN with 33 input nodes and 4 output nodes.

    There are three hidden layers with respectively 64, 128 and 64 nodes. 
    
 This choice is arbitrary.

## dqn_agent.py

The reinforcement agent is specified in this file. The action is chosen from a model that is 
updated periodically every 4 steps.

