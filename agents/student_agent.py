####################### MCTS adapted from https://ai-boson.github.io/mcts/ ##########################
from agents.agent import Agent
from store import register_agent
import sys
import math
import random as random
# import numpy as np
from copy import deepcopy
# import pdb
import time

TWO_SEC_TIME = 1850000000
THIRTY_SEC_TIME = 28500000000

class state():
    def __init__(self, cur_pos, adv_pos, dir, chess_board, max_step):
        self.chess_board = chess_board
        self.cur_pos = cur_pos
        self.adv_pos = adv_pos
        self.dir = dir
        self.max_step = max_step
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    
    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def unset_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = False
        # Set the opposite barrier to False
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = False

    def check_valid_barrier(self, end_pos, barrier_dir):
            r, c = end_pos
            move = self.moves[barrier_dir]
            if self.chess_board[r, c, barrier_dir] or self.chess_board[r + move[0], c + move[1], self.opposites[barrier_dir]]:
                return False
            return True

    def get_legal_actions(self, c_pos, adv_pos):
            legal_actions_queue = list()
            # Use BFS
            state_queue = [(c_pos, 0)]
            visited = {tuple(c_pos)}
            # is_reached = False
            while state_queue:
                cur_pos, cur_step = state_queue.pop(0)

                # Return if the max distance is travelled
                if cur_step == self.max_step+1:
                    break

                # Check if the current location has valid barriers
                for barrier_dir in range(4):
                    if (self.check_valid_barrier(cur_pos, barrier_dir)):
                        legal_actions_queue.append((cur_pos, barrier_dir))

                r, c = tuple(cur_pos)
                
                # Look through the paths of all possible locations (but not in the direction that they came from)
                for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                    if self.chess_board[r, c, dir]:
                        continue
                    a, b = move
                    next_pos = (r+a, c+b)
                    if (next_pos==adv_pos) or tuple(next_pos) in visited:
                        continue
                    visited.add(tuple(next_pos))
                    # print((next_pos, cur_step + 1))
                    state_queue.append((next_pos, cur_step + 1))
            return legal_actions_queue
    
    def manhattan(self, cur_pos, end_pos):
            cr, cc = cur_pos
            er, ec = end_pos
            return (abs(cr - er) + abs(cc-ec))

    def aStar(self, cur_pos, adv_pos, chess_board):

        def neighbours(cur_pos, end_pos, chess_board):
            n = []
            (r, c), g, _ = cur_pos
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if chess_board[r, c, dir]:
                    continue
                a, b = move
                next_pos = (r+a, c+b)
                # print((next_pos, cur_step + 1))
                n.append((next_pos, g+1, self.manhattan((r,c), end_pos)))
            # print("These are the current neighbours: ",n)
            return n

        
        
        #The open and closed sets
        pqueue = set()
        closedset = set()
        #Current point is the starting point
        c_pos = (cur_pos, 0, self.manhattan(cur_pos, adv_pos))
        # print("This is cur: ",c_pos)
        #Add the starting point to the priorityQueue
        pqueue.add(c_pos)
        #While the open set is not empty
        visitedCells = 0
        while pqueue:
            
            #Find the item in the open set with the lowest G + H score
            minF = float('inf')
            c_pos = None
            for cur in pqueue:
                _, g, h = cur
                if minF > g + h:
                    c_pos = cur
                    minF = g + h
            # If it is the item we want, retrace the path and return it

            # If goal is reached, then game has not ended and return
            cur_coord, cur_g, cur_h = c_pos
            # print( "This is cur: ", cur_coord)
            if cur_coord == adv_pos: 
                return False, -1
            pqueue.remove(c_pos) 
            visitedCells += 1   
            closedset.add(cur_coord)
            #Loop through the node's children/siblings
            for next_pos in neighbours(c_pos, adv_pos, chess_board):
                
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
                    next_h = self.manhattan((nr, nc), adv_pos)
                    #Set the parent to our current item
                    # next_pos.parent = cur_pos
                    #Add it to the set
                    pqueue.add(((nr, nc), next_g, next_h))
        #return empty list, as there is not path leading to destination
        return True, visitedCells

    def is_game_over(self):

        return self.aStar(self.cur_pos, self.adv_pos, self.chess_board)

    def game_result(self, occupied):
        # board_size = len(self.chess_board)
        result = None
        # Find the total space occupied by enemy
        _, adv_occupied = self.aStar(self.adv_pos, self.cur_pos, self.chess_board)
        # print("Score: Bot = ", occupied, " Adv = ", adv_occupied)
        if adv_occupied > occupied:
            return -1
        elif adv_occupied - occupied == occupied:
            return 0
        else:
            return 1
        # print("Did I win?", result)
        # return result

    def max_pruning(self, alpha, beta, depth):
        max_value = float('-inf')
        depth+=1
        # isFirst = True
        # result = None

        isOver, occupied = self.is_game_over()
        
        if isOver:
            return self.game_result(occupied)
        # best_action = None
        legal_actions = self.get_legal_actions(self.cur_pos, self.adv_pos)
        # print(legal_actions)
        for max_action in legal_actions:
            # if isFirst:
            #     isFirst = False
            #     best_action = max_action
            (pr, pc) = self.cur_pos
            (cr, cc), dir = max_action
            self.set_barrier(cr,cc,dir)
            self.cur_pos = (cr,cc)
            # pdb.set_trace()
            min_value = self.min_pruning(alpha, beta, depth)
            # Revert changes to the state
            self.unset_barrier(cr,cc,dir)
            self.cur_pos = (pr,pc)
            if min_value > max_value:
                max_value = min_value
                # best_action = max_action

            if max_value >= beta:
                return(max_value)
            if max_value < alpha:
                alpha = max_value
        return max_value



    def min_pruning(self, alpha, beta, depth):
        min_value = float('inf')
        depth += 1
        # result = None
        # isFirst = True
        isOver, occupied = self.is_game_over()
        
        if isOver:
            return self.game_result(occupied)
        # worst_action = None
        if depth > 2:
            return self.simulate_random()
        for min_action in self.get_legal_actions(self.adv_pos, self.cur_pos):
            # if isFirst:
            #     isFirst = False
            #     worst_action = min_action
            (pr, pc) = self.adv_pos
            (cr, cc), dir = min_action
            self.set_barrier(cr,cc,dir)
            self.adv_pos = (cr, cc)
            max_value = self.max_pruning(alpha, beta, depth)
            # Revert changes
            self.unset_barrier(cr,cc,dir)
            self.adv_pos = (pr, pc)
            if max_value < min_value:
                min_value = max_value
                # worst_action = min_action
                
            if min_value <= alpha:
                return(min_value)
            if min_value < beta:
                beta = min_value
        return min_value

    def simulate_random(self):
        isOver , occupied = self.is_game_over()
        while not isOver:
            self = self.simulate_adv()

            isOver, occupied = self.is_game_over()

            if isOver:
                break
            possible_moves = self.get_legal_actions(self.cur_pos, self.adv_pos)
            if len(possible_moves) == 0:
                break
            random.shuffle(possible_moves)
            # 
            closest_action = None
            # pdb.set_trace()
            closest_distance = float('inf')
            for x in possible_moves:
                (xr, xc), x_dir = x

                # Check to make sure there aren't 3 barriers
                wall_no = 0
                for i in range(4):
                    self.set_barrier(xr,xc,x_dir)
                    if (self.chess_board[xr,xc,i]):
                        wall_no += 1
                    self.unset_barrier(xr,xc,x_dir) 
                if wall_no > 2:
                    continue 
                cur_manhattan = self.manhattan((xr,xc), self.adv_pos)
                if closest_distance > cur_manhattan:
                    closest_action = x
                    closest_distance = cur_manhattan
           
            # No legal actions that are less than 3
            if closest_action == None:
                closest_action = possible_moves.pop()



                    
            # pdb.set_trace()
            # list_actions.remove(closest_action)
            # self._untried_actions = set(list_actions)
            # pdb.set_trace()
            


            # # print("Game is over: ", isOver)
            # possible_moves = self.get_legal_actions(self.cur_pos, self.adv_pos)
            # if len(possible_moves) == 0:
            #     break
            # action = possible_moves.pop()
            # print("This is the current action: ",action)
            # if (closest_action == None):
            #     z = possible_moves.pop()
            #     print("This is the current action: ",z)
            #     print(possible_moves)
            #     self.move(z)
            # print("This is the current action: ",closest_action)
            # print(possible_moves)
            self = self.move(closest_action)
            isOver, occupied = self.is_game_over()
        return self.game_result(occupied)

    def move(self, action):
        
        
        # if (action == None):
        #     pdb.set_trace()
        # (r,c), dir = action
        self.set_barrier(r, c, dir)
        # (ar, ac), adir = self.simulate_adv()
        # print("Current adv pos: ", (ar, ac))
        # print("Current cur pos: ", (r, c))
        # self.chess_board[ar, ac, adir] = True
        # self.adv_pos = (ar, ac)
        # self.cur_pos = (r,c)
        # self.dir = dir
        return state((r,c), self.adv_pos, dir, self.chess_board, self.max_step)

    def simulate_adv(self):
        ori_pos = deepcopy(self.adv_pos)
        cur_board = self.chess_board
        legal_moves = self.get_legal_actions(ori_pos, self.cur_pos)
        random.shuffle(legal_moves)
        adv_action = legal_moves.pop()
        (ar, ac), adir = adv_action
        # moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # steps = random.randint(0, self.max_step)
        # # temp value for final position
        # my_pos = self.adv_pos

        # # Random Walk
        # for _ in range(steps):
            
        #     r, c = self.adv_pos
        #     dir = random.randint(0, 3)
        #     m_r, m_c = moves[dir]
        #     my_pos = (r + m_r, c + m_c)

        #     # Special Case enclosed by Adversary
        #     k = 0
        #     while self.chess_board[r, c, dir] or my_pos == self.adv_pos:
        #         k += 1
        #         if k > 300:
        #             break
        #         dir = random.randint(0, 3)
        #         m_r, m_c = moves[dir]
        #         my_pos = (r + m_r, c + m_c)

        #     if k > 300:
        #         my_pos = ori_pos
        #         break

        # # Put Barrier
        # dir = random.randint(0, 3)
        # r, c = my_pos
        # l = 0
        # while cur_board[r, c, dir]:
        #     dir = random.randint(0, 3)
        #     l += 1
        #     if l > 300:
        #         break
        # self.set_barrier(r,c, dir)
        return state(self.cur_pos, (ar, ac), adir, self.chess_board, self.max_step)

class MonteCarloTreeSearchNode():
    def __init__(self, state, time_limit, parent=None, parent_action=None):
            self.state = state
            self.time_limit = time_limit
            self.parent = parent
            self.parent_action = parent_action
            self.children = []
            self._number_of_visits = 0
            self._results = dict()
            self._results[1] = 0
            self._results[0] = 0
            self._results[-1] = 0
            self._untried_actions = None
            self._untried_actions = self.untried_actions()
            return
    def untried_actions(self):

        self._untried_actions = self.state.get_legal_actions(self.state.cur_pos, self.state.adv_pos)
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        ties = self._results[0]
        loses = self._results[-1]
        return wins - loses + 0.5*ties

    def n(self):
        return self._number_of_visits

    def expand(self):
        more_than_3 = list()
        list_actions = self._untried_actions
        random.shuffle(list_actions)
        closest_action = None
        # pdb.set_trace()
        closest_distance = float('inf')
        for x in list_actions:
            (xr, xc), x_dir = x

            # Check to make sure there aren't 3 barriers
            wall_no = 0
            self.state.set_barrier(xr,xc,x_dir)
            for i in range(4):
                if (self.state.chess_board[xr,xc,i]):
                    wall_no += 1
            self.state.unset_barrier(xr,xc,x_dir)    
            if wall_no > 2:
                # more_than_3.append(x)
                continue 

            # Look for moves that are closer to the target
            cur_manhattan = self.state.manhattan((xr, xc), self.state.adv_pos)
            if closest_distance > cur_manhattan:
                closest_action = x
                closest_distance = cur_manhattan
        
        # No legal actions that are less than 3
        if closest_action == None:
            closest_action = list_actions[random.randint(0,len(list_actions)-1)]
            # for x in more_than_3:
            #     (xr, xc), x_dir = x
            #     cur_manhattan = self.state.manhattan((xr, xc), self.state.adv_pos)
            #     if closest_distance < cur_manhattan:
            #         closest_action = x
            #         closest_distance = cur_manhattan
        
        # Update the best action
        # if closest_action == None:
        #     pdb.set_trace()

        list_actions.remove(closest_action)
        # except ValueError as err:
        #     pdb.set_trace()
        # self._untried_actions = set(list_actions)
        # pdb.set_trace()
        (r, c), dir = closest_action
        self.state.set_barrier(r,c,dir)
        next_chess_board = deepcopy(self.state.chess_board)
        self.state.unset_barrier(r,c,dir)
        # next_chess_board = deepcopy(self.state.chess_board)
        # next_chess_board.set_barrier(r,c,dir)
        # print('Function enters here: ')
        child_node = MonteCarloTreeSearchNode(state((r,c), self.state.adv_pos, dir, next_chess_board, self.state.max_step), self.time_limit, parent=self, parent_action=closest_action)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        is_terminal, _ = self.state.is_game_over()
        return is_terminal


    # This default policy needs to be done iterably in one
    def rollout(self):
        # Create copy of chess_board state
        current_rollout_state = deepcopy(self.state)
        # Simulate random game
        # end_state = current_rollout_state.simulate_random()
        
        # value = current_rollout_state.max_pruning()
        # pdb.set_trace()
        # Simulate max_min game
        # (sr,sc) = current_rollout_state.cur_pos()
        # dir = current_rollout_state.dir()
        # current_rollout_state.set_barrier(sr,sc,dir)
        value = current_rollout_state.min_pruning(float('-inf'), float('inf'), 0)
        # if (action == None):
        #     pdb.set_trace()
        # current_rollout_state.unset_barrier(sr,sc,dir)
        # (r,c), dir = action
        # current_rollout_state.set_barrier(r,c,dir)
        # _, occupied = current_rollout_state.is_game_over()
        # print(value)
        # current_rollout_state.simulate_random((current_rollout_state.cur_pos, current_rollout_state.dir))
        return value

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
        
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        
        # Check if the c_param is the right value
        choices_weights = [(c.q() / c.n()) + 0.1*math.sqrt((2 * math.log(self.n()) / c.n())) for c in self.children]
        # Coding np.argmax(choices_weights)
        index = 0
        highest = 0
        for i in range(len(choices_weights)):
            if highest < choices_weights[i]:
                index = i
                highest = choices_weights[i]
        return self.children[index]
        # return self.children[np.argmax(choices_weights)]

    # def rollout_policy(self, possible_moves):
    #     # Apparently pop randomly chooses an element in sets
    #     cur_moves = deepcopy(possible_moves)
    #     return cur_moves.pop()

    def _tree_policy(self):

        current_node = self
        # pdb.set_trace()
        while not current_node.is_terminal_node():
            # print("Does the current node have no more moves? ", current_node.is_fully_expanded())
            # pdb.set_trace()
            if not current_node.is_fully_expanded():
                
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        sim_no = 0
        while time.time_ns() < self.time_limit:
            sim_no += 1
            v = self._tree_policy()
            # pdb.set_trace()
            reward = v.rollout()
            v.backpropagate(reward)
        # print("Total simulations: ", sim_no)
        return self.best_child(c_param=0.1)

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        self.is_first_round = True

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

        # print(self.max_step)
        # First move has 30 seconds while all consequent ones have 2 seconds; initialize root
        if self.is_first_round:
            self.is_first_round = False
            time_limit = time.time_ns() + THIRTY_SEC_TIME
            root = MonteCarloTreeSearchNode(state(self.start_pos, self.adv_pos, -1, deepcopy(self.chess_board), max_step), time_limit)
            selected_node = root.best_action()
        else:
            time_limit = time.time_ns() + TWO_SEC_TIME
            root = MonteCarloTreeSearchNode(state(self.start_pos, self.adv_pos, -1, deepcopy(self.chess_board), max_step), time_limit)
            selected_node = root.best_action()
        # print(selected_node)
        next_pos = selected_node.state.cur_pos
        dir = selected_node.state.dir
        howMuchTime = selected_node.time_limit - time.time_ns()
        print(str(howMuchTime) + " sec")
        return next_pos, dir

    