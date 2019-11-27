import numpy as np
import matplotlib.pyplot as plt
from grid import GAMMA, ALL_POSSIBLE_ACTIONS, SMALL_ENOUGH
from grid import Grid, RandomGrid
from grid import standard_grid, print_values, print_policy, play_game, get_policy_and_value_function
from grid import example_policy, winning_policy



### Iterative Policy Evaluation
### Here the policy is given (deterministic)
### The action state transitions are still deterministic (if you are in state s and you perform action a, you only end up in one possible state s')
def policy_eval(grid, policy):
    states = grid.all_states()
    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0
    gamma = 0.9

    while True:
        delta = 0
        for s in states:
            if not grid.is_terminal(s):
                grid.set_state(s)
                old_v = V[s]
                # In this case, there is only one action per state, so no need for a for loop
                action = policy[s]
                r = grid.move(action)
                v_s_prime = V[grid.current_state()]
                V[s] = r + gamma * v_s_prime
                grid.undo_move(action)

                delta = max(delta, abs(V[s] - old_v))
        if delta < SMALL_ENOUGH:
            break

    return V


### Policy Iteration Algorithm
### Finds the optimal policy, starting with a random policy and ameliorating it
def policy_iteration(g):
    policy = {}

    # Randomly initialize policy
    # Since the policy is a state -> action mapping, randomly initializing the policy is randomly choosing an action for each non terminal state
    for s in g.actions.keys():
        policy[s] = np.random.choice(g.actions[s])

    policy_changed = True

    # Keep updating the policy (by finding better actions for each state) until the policy doesn't change anymore
    while policy_changed:
        # 1. POLICY EVALUATION
        V = policy_eval(g, policy)

        policy_changed = False
        gamma = 0.9
        states = g.all_states()

        # 2. POLICY IMPROVEMENT
        for s in states:
            if not g.is_terminal(s):
                g.set_state(s)
                old_a = policy[s]  # Old action determined by the old policy

                # Find the best action (action with highest value)
                max_a = old_a
                max_val = float('-inf')
                for a in g.actions[s]:
                    # Perform action, get value, then undo action
                    r = g.move(a)
                    val = r + gamma * V[g.current_state()]
                    if val > max_val:
                        max_a = a
                        max_val = val
                    g.undo_move(a)
                policy[s] = max_a

                # if the new action is different from the one determined by the old policy, then the policy has changed
                if policy[s] != old_a:
                    policy_changed = True

    return policy, V

grid = standard_grid()
policy = winning_policy
V = policy_eval(grid, policy)
print_values(V, grid)
print_policy(policy, grid)