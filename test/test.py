import math
import numpy as np
from copy import deepcopy

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = dict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

def untried_actions(self):

    self._untried_actions = self.state.get_legal_actions()
    return self._untried_actions

def q(self):
    wins = self._results[1]
    loses = self._results[-1]
    return wins - loses

def n(self):
    return self._number_of_visits

def expand(self):
	
    action = self._untried_actions.pop()
    next_state = self.state.move(action)
    child_node = MonteCarloTreeSearchNode(
		next_state, parent=self, parent_action=action)

    self.children.append(child_node)
    return child_node 

def is_terminal_node(self):
    return self.state.is_game_over()

def rollout(self):
    current_rollout_state = self.state
    isOver, occupied = current_rollout_state.is_game_over()
    while not isOver:
        
        possible_moves = current_rollout_state.get_legal_actions()
        
        action = self.rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
    return current_rollout_state.game_result(occupied)

def backpropagate(self, result):
    self._number_of_visits += 1.
    self._results[result] += 1.
    if self.parent:
        self.parent.backpropagate(result)
    
def is_fully_expanded(self):
    return len(self._untried_actions) == 0

def best_child(self, c_param=0.1):
    
    # Check if the c_param is the right value
    choices_weights = [(c.q() / c.n()) + math.sqrt((2 * math.log(self.n()) / c.n())) for c in self.children]
    return self.children[np.argmax(choices_weights)]

def rollout_policy(self, possible_moves):
    
    return possible_moves[math.random.randint(len(possible_moves))]

def _tree_policy(self):

    current_node = self
    while not current_node.is_terminal_node():
        
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node

def best_action(self):
    simulation_no = 100
	
	
    for i in range(simulation_no):
		
        v = self._tree_policy()
        reward = v.rollout()
        v.backpropagate(reward)
	
    return self.best_child(c_param=0.)

# def main():
#     root = MonteCarloTreeSearchNode(state = initial_state)
#     selected_node = root.best_action()
#     return 

def check_valid_barrier(self, end_pos, barrier_dir):
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        return True

def get_legal_actions(self):
        legal_actions_queue = set()
        # Use BFS
        state_queue = [(self.start_pos, 0)]
        visited = {tuple(self.start_pos)}
        is_reached = False
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)

            # Return if there max distance is travelled
            if cur_step == self.max_step+1:
                            break

            # Check if the current location has valid barriers
            for barrier_dir in range(4):
                if (self.check_valid_barrier(cur_pos, barrier_dir)):
                    legal_actions_queue.add((cur_pos, barrier_dir))

            r, c = tuple(cur_pos)
            
            # Look through the paths of all possible locations (but not in the direction that they came from)
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if self.chess_board[r, c, dir]:
                    continue
                a, b = move
                next_pos = (r+a, c+b)
                if (next_pos==self.adv_pos) or tuple(next_pos) in visited:
                    continue
                visited.add(tuple(next_pos))
                # print((next_pos, cur_step + 1))
                state_queue.append((next_pos, cur_step + 1))
        return legal_actions_queue


 


# def game_result(self):

# def is_Connected(self, barrier):

#     r,c = self.start_pos
        
#     if (r == 0):
#         if (barrier == 1):
#             return self.chess_board[r+1,c, 1]
#     if
