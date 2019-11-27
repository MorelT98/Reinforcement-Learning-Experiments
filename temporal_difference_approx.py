import numpy as np
import matplotlib.pyplot as plt
from grid import GAMMA, RandomGrid
from grid import example_policy, winning_policy
from grid import ALL_POSSIBLE_ACTIONS
from grid import standard_grid, print_values, print_policy, play_game, get_policy_and_value_function
from temporal_difference import td_0_prediction, sarsa, C, epsilon_greedy_from
from monte_carlo_approx import feature_vector, get_value_function
ALPHA = 0.0001
EPSILON = 0.05

# grid = standard_grid(random=True)
# policy = example_policy
# V = td_0_prediction(grid, policy, 10000)
# print_values(V, grid)
# print_policy(policy, grid)



def td_0_approx_prediction(grid, policy, N = 1000, verbose = False):
    theta = np.random.random(4)
    errors = np.zeros(N)
    T = 1
    for t in range(N):
        if t % 100 == 0:
            T += 0.8
        alpha = ALPHA / T
        s = grid.official_start

        grid.set_state(s)    # Always reset the start position back to official start!

        if verbose: print('Game #', t + 1)

        # Instead of using while not game_over(), I only allowed the game
        #    to go on width * height steps to prevent the game from going
        #    forever sometimes
        error = 0
        for i in range(grid.width * grid.height):
            old_state = grid.current_state()

            old_x = feature_vector(old_state)

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
            x = feature_vector(s)

            if grid.is_terminal(s):
                target = r
            else:
                target = r + GAMMA * theta.dot(x)

            # Update value of V(s), given the equation
            #    V(s) <- V(s) + alpha[r + gamma * V(s') - V(s)]
            theta = theta + alpha * (target - theta.dot(old_x)) * old_x

            v = V[old_state]
            v_approx = theta.dot(old_x)
            error += (v - v_approx) ** 2

            if grid.game_over():
                break
        error /= i
        errors[t] = error

    plt.plot(errors)
    plt.show()

    return theta


# theta = td_0_approx_prediction(grid, policy, 20000)
# V = get_value_function(grid, theta)
# print_values(V, grid)


# ---------------------------------------------------------------------------------------------
# SARSA
grid = standard_grid()
Q = sarsa(grid, 10000)
policy, V = get_policy_and_value_function(grid, Q)
print_values(V, grid)
print_policy(policy, grid)

all_sa = []
for s in grid.actions.keys():
    for a in ALL_POSSIBLE_ACTIONS:
        all_sa.append((s, a))


def one_hot(s, a):
    x = np.zeros(36)
    i = all_sa.index((s, a))
    x[i] = 1
    return x

def feature_vector(s, a):
    x = np.zeros(28)
    x[0] = 1
    i, j = s
    if a == 'U':
        t = 1
    elif a == 'D':
        t = 8
    elif a == 'L':
        t = 14
    else:
        t = 20
    x[t:t+7] = [1, i - 1, j - 1.5, i * i - 5/3, j * j - 3.5, i * j - 1.5, (i * j) ** 2 - 35/6]
    return x


def get_q_dict(theta, s, transformation):
    Q = {}
    for a in ALL_POSSIBLE_ACTIONS:
        x = transformation(s, a)
        q = theta.dot(x)
        Q[(s, a)] = q
    return Q

def sarsa_approx(grid, N, tranformation = one_hot):
    if tranformation == one_hot:
        size = 36
    else:
        size = 28

    # Random initialization of the parameters
    theta = np.random.random(size)

    # Average error at each iteration, for debugging purposes
    errors = np.zeros(N)

    T_eps = 1   # Used to decrease epsilon
    T_alpha = 1    # Used to decrease the learning rate
    for t in range(N):

        if t % 100 == 0:
            T_alpha += 0.1
            T_eps += 0.001

        alpha = ALPHA / T_alpha
        eps = EPSILON / T_eps

        s = grid.official_start
        grid.set_state(s)

        # Get next action
        q = get_q_dict(theta, s, tranformation)
        a = epsilon_greedy_from(q, s, eps)

        # Play game
        error = 0
        count = 0    # Number of steps in the game
        for i in range(grid.height * grid.width):
            count += 1
            # Take action
            r = grid.move(a)
            s_prime = grid.current_state()

            # Give negative reward if illegal action
            if s_prime == s:
                r = -100

            # Update theta
            if grid.is_terminal(s_prime):
                target = r
            else:
                q_prime = get_q_dict(theta, s_prime, tranformation)
                a_prime = epsilon_greedy_from(q_prime, s_prime, eps)
                x_prime = tranformation(s_prime, a_prime)
                approx_prime = theta.dot(x_prime)
                target = r + GAMMA * approx_prime

            x = tranformation(s, a)
            approx = theta.dot(x)

            theta = theta + alpha * (target - approx) * x

            error = (approx - Q[(s, a)]) ** 2


            # End the game
            if grid.game_over():
                break

            # Update states and actions
            s = s_prime
            a = a_prime

        error /= count
        errors[t] = error

    plt.plot(errors)
    plt.show()

    return theta

# for s in grid.all_states():
#     seen = set()
#     for a in ALL_POSSIBLE_ACTIONS:
#         x = feature_vector(s, a)
#         print(x)


theta = sarsa_approx(grid, 20000, tranformation=feature_vector)

Q_approx = {}
for s, a in all_sa:
    if not grid.is_terminal(s):
        x = feature_vector(s, a)
        Q_approx[(s, a)] = theta.dot(x)

policy, V = get_policy_and_value_function(grid, Q_approx)
print_values(V, grid)
print_policy(policy, grid)
#
# s = (0, 2)
# a = 'R'
# x = one_hot(s, a)
# print(x)
# q = theta.dot(x)
# print(q)