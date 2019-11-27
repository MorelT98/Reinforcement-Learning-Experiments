import numpy as np
import matplotlib.pyplot as plt
from grid import GAMMA, ALL_POSSIBLE_ACTIONS, EPSILON
from grid import Grid, RandomGrid
from grid import standard_grid, print_values, max_action_value, get_policy_and_value_function, print_policy
from grid import example_policy, winning_policy
ALPHA = 0.1
C = 0.1

def td_0_prediction(grid, policy, N = 1000, verbose = False):
    V = {}

    for t in range(1, N + 1):
        s = grid.official_start

        grid.set_state(s)    # Always reset the start position back to official start!

        if verbose: print('Game #', t)

        # Instead of using while not game_over(), I only allowed the game
        #    to go on width * height steps to prevent the game from going
        #    forever sometimes
        for i in range(grid.width * grid.height):
            old_state = grid.current_state()

            # Previous value of V(s)
            old_v = V.get(old_state, 0)

            rand = np.random.random()

            # Epsilon greedy
            if rand < EPSILON:
                a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            else:
                a = policy[s]

            # Take action
            if isinstance(grid, RandomGrid):
                r, a2 = grid.move_randomly(a)
            else:
                r = grid.move(a)
            s = grid.current_state()

            # Update value of V(s), given the equation
            #    V(s) <- V(s) + alpha[r + gamma * V(s') - V(s)]
            new_v = old_v + ALPHA * (r + GAMMA * V.get(s, 0) - old_v)
            V[old_state] = new_v

            if grid.game_over():
                break
    return V


# grid = standard_grid()
# policy = winning_policy
# V = td_0_prediction(grid, policy, 10000)
# print_values(V, grid)


# Given the action value function Q, and the current state s,
#    this function uses epsilon greedy to determine the next
#    action
def epsilon_greedy_from(Q, s, eps):
    r = np.random.random()
    if r < eps:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        # Get the action that has the maximum value
        return max_action_value(Q, s)[0]

def sarsa(grid, N):
    # Random initialization of Q
    Q = {}
    count = {}  # Keeps track of how many times a state action pair (s, a) was involved
    for s in grid.all_states():
        for a in ALL_POSSIBLE_ACTIONS:
            count[(s, a)] = 1
            Q[(s, a)] = 0

    T = 1
    for t in range(1, N + 1):
        # Increase T after every 100 games, to eventually decrease epsilon
        if t % 100 == 0:
            T += 10e-3
        eps = C / T

        # Always start at the official start state
        s = grid.official_start
        grid.set_state(s)

        # Get next action
        a = epsilon_greedy_from(Q, s, eps)

        unecessary_action = False
        # print('Game', t)
        for i in range(grid.height * grid.width):
            #grid.draw_grid()
            #print('action:', a)
            #print()

            # Decrease the learning rate to avoid bouncing off Q*
            alpha = ALPHA / count[(s, a)]
            count[(s, a)] += 1

            # Take action
            r = grid.move(a)
            s_prime = grid.current_state()

            # If the action was invalid, give a very negative reward
            #    to prevent the player to take that action again
            if s_prime == s:
                #unecessary_action = True
                r = -100

            # Get next action now, necessary for calculating Q(s, a)
            a_prime = epsilon_greedy_from(Q, s_prime, eps)

            if grid.is_terminal(s_prime):
                Q[(s_prime, a_prime)] = 0

            # Update Q(s, a) using formula:
            #    Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
            Q[(s, a)] = Q[(s, a)] + alpha * (r + GAMMA * Q[(s_prime, a_prime)] - Q[(s, a)])

            if grid.game_over() or unecessary_action:
                break

            s = s_prime
            a = a_prime
        # print('Game Over')
        # print('------------------------------------')

    return Q

# grid = standard_grid()
# Q = sarsa(grid, 10000)
# policy , V= get_policy_and_value_function(grid, Q)
# print_values(V, grid)
# print_policy(policy, grid)

# Very similar to sarsa, except that it is off policy, meaning that we don't choose action based
#    on suboptimal policy, we just choose a random action all the time.
# However, to update Q(s, a), we use the formula:
#    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a'_Q(s', a') - Q(s, a))
def Q_learning(grid, N):
    # Random initialization of Q
    Q = {}
    count = {}
    for s in grid.all_states():
        for a in ALL_POSSIBLE_ACTIONS:
            count[(s, a)] = 1
            Q[(s, a)] = 0

    T = 1
    for t in range(1, N + 1):

        # Always start at the official start state
        s = grid.official_start
        grid.set_state(s)

        # Next action is always random
        a = np.random.choice(ALL_POSSIBLE_ACTIONS)

        # print('Game', t)
        while not grid.game_over():
            # grid.draw_grid()
            # print('action:', a)
            # print()

            # Decrease the learning rate to avoid bouncing off Q*
            alpha = ALPHA / count[(s, a)]
            count[(s, a)] += 1

            # take action
            r = grid.move(a)
            s_prime = grid.current_state()

            # If action was illegal, the player gets a big negative reward
            #if s_prime == s:
            #    r = -100

            # Next action is always random
            a_prime = np.random.choice(ALL_POSSIBLE_ACTIONS)

            # Get max_a'_Q(s', a')
            max_val = max_action_value(Q, s_prime)[1]

            # Use formula to get Q[(s, a)]
            Q[(s, a)] = Q[(s, a)] + alpha * (r + GAMMA * max_val - Q[(s, a)])

            s = s_prime
            a = a_prime
        # print('Game Over')
        # print('------------------------------------')

    return Q


# grid = standard_grid(step_cost=-0.1)
#
# Q = Q_learning(grid, 10000)
#
# policy, V = get_policy_and_value_function(grid, Q)
# print_values(V, grid)
# print_policy(policy, grid)