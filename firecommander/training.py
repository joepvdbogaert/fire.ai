"""This module contains functions for training agents on environments and for visualizing
the training process.
"""
def train(agent, env, n):
    """ Train an agent on an environment. Basic utility, intended to be extended. """
    # define first action and state
    action = ("NONE", "NONE", "NONE")
    state = env.extract_state()
    total_reward = 0

    # "training" loop
    for _ in range(n):

        # take an action and receive the results   
        action = agent.choose_action(state, env.get_legal_actions())
        next_state, reward = env.step(action)

        # learn from this experience
        agent.train(state, action, reward, next_state)

        # update stuff
        total_reward += reward
        state = next_state

    return total_reward
