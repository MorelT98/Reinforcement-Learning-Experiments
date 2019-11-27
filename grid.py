import numpy as np
SMALL_ENOUGH = 10e-4    # threshold for convergence
GAMMA = 0.9    # discount factor
ALL_POSSIBLE_ACTIONS = ['U', 'D', 'L', 'R']
EPSILON = 0.1

class Grid:  # Environment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.official_start = start
        self.i = start[0]
        self.j = start[1]
        self.rewards = {}
        self.actions = {}

    def set(self, rewards, actions):
        # rewards should be a dictionary of: (i, j): r, or: (row, col): reward
        # actions should be a dictionary of: (i, j): A or (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        # The state s is the location of the player in the grid: s = (i, j)
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        # A terminal state won't be in the actions dictionary (since it won't have any associated action)
        return s not in self.actions

    def move(self, action):
        # check if legal move first
        # Possible actions: U/D/L/R
        if action in self.actions[self.current_state()]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return reward (if any)
        return self.rewards.get(self.current_state(), 0)

    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert (self.current_state() in self.all_states())

    def game_over(self):
        # The game is over if we are in a state where no action is possible
        return self.current_state() not in self.actions

    def all_states(self):
        # Cast to a set to avoid repetition in states
        return set(list(self.rewards.keys()) + list(self.actions.keys()))

    def draw_grid(self):
        states = self.all_states()
        for i in range(self.height):
            for j in range(self.width):
                s = (i, j)
                if s in states:
                    if self.current_state() == s:
                        symbol = 's'
                    else:
                        symbol = '.'
                else:
                    symbol = 'x'
                print(symbol, end='')
                if j != self.width - 1:
                    print('   ', end='')
            print('\n')



def standard_grid(step_cost = 0, random = False):
    # Define a grid that describes the reward for arriving at each state
    #     and possible actions at each state
    # The grid looks like this
    # x means you can't go there
    # s means current position
    # number means reward at that state
    # .   .   .   1
    # .   x   .  -1
    # s   .   .   .
    if random:
        g = RandomGrid(4, 3, (2, 0))
    else:
        g = Grid(4, 3, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -100}
    actions = {
        (0,0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('D', 'L', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('U', 'L', 'R'),
        (2, 3): ('U', 'L')
    }
    g.set(rewards, actions)
    for s in g.all_states():
        if not g.is_terminal(s):
            g.rewards[s] = step_cost
    return g


class RandomGrid(Grid):
    def move_randomly(self, action):
        remaining_actions = list(ALL_POSSIBLE_ACTIONS)
        r = np.random.random()
        # Do the given action with a probability of 0.5
        if r > 0.5:
            return self.move(action), action
        else:
            # Do one of the other actions, each with a
            #  probability of 0.5 / n (In this case, 0.5 / 3)
            remaining_actions.remove(action)
            n = len(remaining_actions)
            p = 0.5 / n
            for i in range(n):
                if i * p <= r < (i + 1) * p:
                    action = remaining_actions[i]
                    break
            return self.move(action), action



# V is the value function dictionary, and g is the grid (environment)
def print_values(V, g):
    for i in range(g.height):
        print('-------------------------')
        print('|', end = "")
        for j in range(g.width):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end = "")
            else:
                print("%.2f|" % v, end = "")    # negative sign takes up an extra space
        print()
    print('-------------------------')


# P is the policy dictionary (Mapping each state to the action to take)
def print_policy(P, g):
    for i in range(g.height):
        print('-------------------------')
        print('|', end = '')
        for j in range(g.width):
            a = P.get((i, j), ' ')
            print('  %s  |' % a, end = '')
        print()
    print('-------------------------')


def play_game(grid, policy, start = (None, None), epsilon_greedy = False, verbose = False):

    if start[0] is None:
        # If using epsilon greedy, start_state = official_start
        if epsilon_greedy:
            start_state = grid.official_start
        # If not, then we use explore starts method: chose a start state randomly
        else:
            possible_starts = list(grid.actions.keys())
            start_state_idx = np.random.choice(len(possible_starts))
            start_state = possible_starts[start_state_idx]
    else:
        start_state = start[0]


    if start[1] is None:
        # If using epsilon greedy, start_action is the one specified by policy
        if epsilon_greedy:
            a = policy[start_state]
        else:
            # If not, then we use explore starts method: chose a start action randomly
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        a = start[1]

    grid.set_state(start_state)
    s = grid.current_state()

    states_actions_rewards = [(s, a, 0)]
    # In any other case, just follow the policy until game is over
    for i in range(grid.width * grid.height):
        if verbose: grid.draw_grid()
        if verbose: print('action:', a)

        old_s = grid.current_state()

        if isinstance(grid, RandomGrid):
            r, a2 = grid.move_randomly(a)
        else:
            r = grid.move(a)

        s = grid.current_state()

        # If the action performed made the game return to an old state,
        #    end the game and give a very negative reward
        #if old_s == s:
        #    states_actions_rewards.append((s, None, -100))
        #    break
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            if epsilon_greedy:
                rand = np.random.random()
                # Choose a random action with probability epsilon
                if rand < EPSILON:
                    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
                else:
                # Follow policy with probability 1 - epsilon
                    a = policy[s]
            else:
                # If not using epsilon greedy, just follow policy
                a = policy[s]
            states_actions_rewards.append((s, a, r))


    if verbose: grid.draw_grid()
    if verbose: print('Game over')
    if verbose: print('------------------------')

    G = 0
    first = True
    states_actions_returns = []
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G

    states_actions_returns.reverse()

    if verbose: print(states_actions_returns)
    if verbose: print('------------------------')
    return states_actions_returns

def get_policy_and_value_function(grid, Q, old_policy = None):
    policy = {}
    value_function = {}
    if old_policy is None:
        for s in grid.all_states():
            if not grid.is_terminal(s):
                a_idx = np.random.choice(len(ALL_POSSIBLE_ACTIONS))
                policy[s] = ALL_POSSIBLE_ACTIONS[a_idx]
    else:
        policy = old_policy


    for s, a in Q.keys():
        if s not in value_function:
            if not grid.is_terminal(s):
                value_function[s] = Q[(s, a)]
                policy[s] = a
            else:
                value_function[s] = 0
        else:
            if Q[(s, a)] > value_function[s]:
                value_function[s] = Q[(s, a)]
                policy[s] = a

    return policy, value_function

def get_value_function_from_policy_and_Q(grid, policy, Q):
    V = {}
    for s in grid.all_states():
        if grid.is_terminal(s):
            V[s] = 0
        else:
            a = policy[s]
            V[s] = Q[(s, a)]
    return V

def max_action_value(Q, s):
    max_sa = None
    max_val = float('-inf')
    for sa in Q.keys():
        if sa[0] == s:
            if Q[sa] > max_val:
                max_val = Q[sa]
                max_sa = sa

    return max_sa[1], max_val

example_policy = {
    (0,0):'R',
    (0, 1):'R',
    (0, 2):'R',
    (1, 0):'U',
    (1, 2):'R',
    (2, 0):'U',
    (2, 1):'R',
    (2, 2):'R',
    (2, 3):'U'
}

winning_policy = {
    (0,0):'R',
    (0, 1): 'R',
    (0, 2):'R',
    (1, 0):'U',
    (1, 2):'U',
    (2, 0):'U',
    (2, 1):'R',
    (2, 2):'U',
    (2, 3):'L'
}

#g = standard_grid(step_cost=-0.1, random=False)
#play_game(g, winning_policy, epsilon_greedy=True, epsilon= 0.1, verbose = True)




