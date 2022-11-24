# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import math
import numpy as np
from copy import deepcopy


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.start_pos = my_pos
        self.adv_pos = adv_pos
        self.chess_board = chess_board
        self.max_step = max_step
        self.next_pos = my_pos


        root = MonteCarloTreeSearchNode(deepcopy(self.chess_board))
        selected_node = root.best_action()
        print(selected_node)

        # dummy return
        return my_pos, self.dir_map["u"]

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
        return self.state.is_game_over(self.state)

    def rollout(self):
        current_rollout_state = self.state
        isOver, occupied = current_rollout_state.is_game_over(self.state)
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
    
    def aStar(self, chess_board):

        def neighbours(cur_pos, end_pos, chess_board):
            n = []
            (r, c), g, _ = cur_pos
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if chess_board[r, c, dir]:
                    continue
                a, b = move
                next_pos = (r+a, c+b)
                # print((next_pos, cur_step + 1))
                n.append((next_pos, g+1, manhattan((r,c), end_pos)))
            # print("These are the current neighbours: ",n)
            return n

        def manhattan(cur_pos, end_pos):
            cr, cc = cur_pos
            er, ec = self.adv_pos
            return (abs(cr - er) + abs(cc-ec))
        
        #The open and closed sets
        pqueue = set()
        closedset = set()
        #Current point is the starting point
        cur_pos = (self.start_pos, 0, manhattan(self.start_pos, self.adv_pos))
        # print("This is cur: ",cur_pos)
        #Add the starting point to the priorityQueue
        pqueue.add(cur_pos)
        #While the open set is not empty
        visitedCells = 0
        while pqueue:
            
            #Find the item in the open set with the lowest G + H score
            minF = float('inf')
            cur_pos
            for cur in pqueue:
                _, g, h = cur
                if minF > g + h:
                    cur_pos = cur
                    minF = g + h
            # If it is the item we want, retrace the path and return it

            # If goal is reached, then game has not ended and return
            cur_coord, cur_g, cur_h = cur_pos
            print( "This is cur: ", cur_coord)
            if cur_coord == self.adv_pos: 
                return False, -1
            pqueue.remove(cur_pos) 
            visitedCells += 1   
            closedset.add(cur_coord)
            #Loop through the node's children/siblings
            for next_pos in neighbours(cur_pos, self.adv_pos, chess_board):
                
                (nr, nc), next_g, next_h = next_pos
                #If it is already in the closed set, skip it
                if (nr, nc) in closedset:
                    continue
                #Otherwise if it is already in the open set
                if next_pos in pqueue:
                    #Check if we beat the G score 
                    
                    new_g = cur_g + 1
                    if next_g > new_g:
                        #If so, update the node to have a new parent
                        next_g = new_g
                        # next_pos.parent = cur_pos
                else:
                    #If it isn't in the open set, calculate the G and H score for the node
                    next_g = cur_g + 1
                    next_h = manhattan((nr, nc), self.adv_pos)
                    #Set the parent to our current item
                    # next_pos.parent = cur_pos
                    #Add it to the set
                    pqueue.add(((nr, nc), next_g, next_h))
        #return empty list, as there is not path leading to destination
        return True, visitedCells

    def is_game_over(self):
        
        return self.aStar(self.state)

    def game_result(self, occupied):
        board_size = len(self.chess_board)
        if board_size*board_size - occupied > occupied:
            return -1
        elif board_size*board_size - occupied == occupied:
            return 0
        else:
            return 1

    def move(self, action):
        (r,c), dir = action
        chessCopy = deepcopy(self.chess_board)
        chessCopy[r, c, self.dir_map[dir]] = True
        return chessCopy

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

    