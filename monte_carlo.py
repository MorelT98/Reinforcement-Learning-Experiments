import numpy as np
import matplotlib.pyplot as plt
from grid import GAMMA, ALL_POSSIBLE_ACTIONS, SMALL_ENOUGH
from grid import Grid, RandomGrid
from grid import standard_grid, print_values, print_policy, play_game, \
    get_policy_and_value_function, get_value_function_from_policy_and_Q
from grid import example_policy, winning_policy

def first_visit_monte_carlo_prediction(grid, policy, N):
    Q = {}
    all_returns = {}
    for s in grid.all_states():
        if not grid.is_terminal(s):
            for a in ALL_POSSIBLE_ACTIONS:
                all_returns[(s, a)] = []

    for i in range(N):
        states_actions_returns = play_game(grid, policy)
        seen = set()

        for s, a, g in states_actions_returns:
            if not (s, a) in seen:
                all_returns[(s, a)].append(g)
                Q[(s, a)] = np.mean(all_returns[(s, a)])
                seen.add((s, a))


    V = get_value_function_from_policy_and_Q(grid, policy, Q)

    return V

#g = standard_grid(step_cost=0, random=True)
#policy = winning_policy
#V = first_visit_monte_carlo_prediction(g, policy, 1000)
#print_values(V, g)
#print_policy(policy, g)


def monte_carlo_control(grid, N, exploring_starts = False, epsilon = 0.1, verbose = False):
    # Initialize a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # Initialize Q(s, a) and returns
    Q = {}
    all_returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            for a in ALL_POSSIBLE_ACTIONS:
                Q[(s, a)] = 0
                all_returns[(s, a)] = []
        else:
            pass


    deltas = []
    for t in range(N):
        if verbose:
            if t % (N // 100) == 0:
                print(t)
                print_policy(policy, grid)
        delta = 0.0
        states_actions_returns = play_game(grid, policy, verbose=False)
        seen = set()
        for s, a, g in states_actions_returns:
            if (s, a) not in seen:
                old_q = Q[(s, a)]
                all_returns[(s, a)].append(g)
                Q[(s, a)] = np.mean(all_returns[(s, a)])
                diff = abs(old_q - Q[(s, a)])
                delta = max(delta, diff)
                seen.add((s, a))
        deltas.append(delta)

        policy, V = get_policy_and_value_function(grid, Q, policy)

    plt.plot(deltas)
    #plt.show()
    #print_values(V, grid)
    return policy, V

# g = standard_grid(step_cost=-0.1, random=True)
#
# print('Rewards')
# print_values(g.rewards, g)
#
# policy, V = monte_carlo_control(g, 3000)
#
# print_policy(policy, g)
# print_values(V, g)