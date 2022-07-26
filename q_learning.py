import argparse
import numpy as np
from environment import MountainCar, GridWorld
import random
import matplotlib.pyplot as plt


# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
# - determining which state to visit next

def sparse2arr(state, mode, state_space):
    s = np.zeros(state_space)
    if mode == "raw":
        s[0] = state[0]
        s[1] = state[1]
    else:
        for key, value in state.items():
            s[key] = value
    return s

def Q(s, a, w, b):
    return np.dot(s, w[a]) + b

def find_best_action(s, weight, bias, env):
    best_action = -1
    best_q = -(10**31)
    for i in range(env.action_space):
        if Q(s, i, weight, bias) > best_q:
            best_action = i
            best_q = Q(s, i, weight, bias)
    return best_action

def main(args):
    # Command line inputs
    mode = 'tile' #args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = 400 #args.episodes
    max_iterations = 200 #args.max_iterations
    epsilon = 0.05 #args.epsilon
    gamma = 0.99 #args.gamma
    learning_rate = 0.00005 #args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    # QUESTIONS: WHAT IS A(4?) AND WHAT IS S(12?)?
    #            WHAT STATE IS THE TERMINAL STATE?
    # Env.step gives new state, reward, bool
    # TODO: Initialize your weights/bias here
    # weights = ...  # Our shape is |A| x |S|, if this helps. 
    # bias = ...
    # If you decide to fold in the bias (hint: don't), recall how the bias is
    # defined!
    weight = np.zeros((env.action_space, env.state_space))
    bias = 0

    returns = []  # This is where you will save the return after each episode
    for episode in range(episodes):
        # Reset the environment at the start of each episode
        state = env.reset()  # `state` now is the initial state
        reward_sum = 0
        for it in range(max_iterations):
            # TODO: Fill in what we have to do every iteration
            # Hint 1: `env.step(ACTION)` makes the agent take an action
            #         corresponding to `ACTION` (MUST be an INTEGER)
            # Hint 2: The size of the action space is `env.action_space`, and
            #         the size of the state space is `env.state_space`
            # Hint 3: `ACTION` should be one of 0, 1, ..., env.action_space - 1
            # Hint 4: For Grid World, the action mapping is
            #         {"up": 0, "down": 1, "left": 2, "right": 3}
            #         Remember when you call `env.step()` you have to pass
            #         the INTEGER representing each action!
            s = sparse2arr(state, mode, env.state_space)
            if random.random() > epsilon:
                # take greedy action
                action = find_best_action(s, weight, bias, env)
            else:
                action = random.randrange(0, env.action_space, 1)

            new_state, reward, done = env.step(action) 
            reward_sum += reward

            new_s = sparse2arr(new_state, mode, env.state_space)
            new_action = find_best_action(new_s, weight, bias, env)
            old_weight = weight
            x = learning_rate*(Q(s, action, weight, bias) - (reward + gamma*Q(new_s, new_action, weight, bias)))
            temp = np.dot(x, s)
            weight[action] -= temp
            bias -= x
            state = new_state
            if done == True:
                break
        returns.append(reward_sum)
    res = np.hstack((np.array(bias), weight.flatten('F')))
    # TODO: Save output files
    np.savetxt(weight_out, res)
    np.savetxt(returns_out, returns)
    mean = []
    ret = np.array(returns)
    for i in range(0, ret.shape[0]):
        if i < 25:
            try: 
                mean.append(np.sum(ret[0:i])/i)
            except:
                print(i)
                print(ret)
        else:
            mean.append(np.sum(ret[i-25:i])/25) 
        
    x = list(range(0,episode+1)) 
    # y = np.array([10, 20, 50, 100, 200])
    plt.plot(x, returns, label='returns')
    plt.plot(x, mean, label='mean')
    # plt.plot(x, valid_nll_arr, label='validation')
    plt.legend()
    plt.xlabel('episodes')
    plt.ylabel('returns')
    plt.show()

if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    main(parser.parse_args())


