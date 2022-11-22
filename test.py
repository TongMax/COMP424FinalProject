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
    
    while not current_rollout_state.is_game_over():
        
        possible_moves = current_rollout_state.get_legal_actions()
        
        action = self.rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
    return current_rollout_state.game_result()

def backpropagate(self, result):
    self._number_of_visits += 1.
    self._results[result] += 1.
    if self.parent:
        self.parent.backpropagate(result)
    
def is_fully_expanded(self):
    return len(self._untried_actions) == 0

def best_child(self, c_param=0.1):
    
    choices_weights = [(c.q() / c.n()) + c_param * math.sqrt((2 * math.log(self.n()) / c.n())) for c in self.children]
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

def check_valid_step(self, start_pos, end_pos, adv_pos, barrier_dir, chess_board, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if (start_pos==end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            
            if cur_step == max_step:
                break
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if (next_pos==adv_pos).all() or tuple(next_pos) in visited:
                    continue
                if (next_pos==end_pos).all():
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached


def get_legal_actions(self): 
    legal_actions_queue = [(self.start_pos, 0)]



# def is_game_over(self):

# def game_result(self):

# def move(self,action):