import numpy as np
import matplotlib.pyplot as plt
from grid import example_policy, winning_policy
from grid import standard_grid, print_values, print_policy, play_game
from monte_carlo import first_visit_monte_carlo_prediction
ALPHA = 0.2

# grid = standard_grid(step_cost=-0.1, random=False)
# policy = winning_policy
# V = first_visit_monte_carlo_prediction(grid, policy, 1000)
# print_values(V, grid)
#print_policy(policy, grid)

def feature_vector(s):
    return np.array([1, s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3])

def mc_approx_prediction(grid, policy, N):
    theta = np.random.random(4)
    errors = np.zeros(N)
    deltas = []

    t = 1
    for i in range(N):
        if i % 100 == 0:
            t += 0.01
        alpha = ALPHA / t
        delta = 0
        states_actions_returns = play_game(grid, policy)
        seen = set()
        count = 0
        for s, a, g in states_actions_returns:
            if s not in seen:
                x = feature_vector(s)
                old_theta = theta.copy()
                theta = theta + alpha * (g - theta.dot(x)) * x

                errors[i] += (g - theta.dot(x)) ** 2
                count += 1
                delta = max(delta, np.abs((old_theta - theta)).sum())
                seen.add(s)
        errors[i] /= count
        deltas.append(delta)

    plt.plot(errors)
    plt.show()

    return theta

def get_value_function(grid, theta):
    V = {}
    for s in grid.all_states():
        if grid.is_terminal(s):
            V[s] = 0
        else:
            x = feature_vector(s)
            V[s] = theta.dot(x)
    return V

# theta = mc_approx_prediction(grid, policy, 50000)
# V = get_value_function(grid, theta)
# print_values(V, grid)