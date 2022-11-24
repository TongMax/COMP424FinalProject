# Human Input agent
from agents.agent import Agent
from store import register_agent
import sys
import time
from copy import deepcopy


@register_agent("human_agent")
class HumanAgent(Agent):
    def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.start_pos = my_pos
        self.adv_pos = adv_pos
        self.chess_board = chess_board
        self.max_step = max_step
        text = input("Your move (x,y,dir) or input q to quit: ")
        while len(text.split(",")) != 3 and "q" not in text.lower():
            print("Wrong Input Format!")
            text = input("Your move (x,y,dir) or input q to quit: ")
        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)
        x, y, dir = text.split(",")
        x, y, dir = x.strip(), y.strip(), dir.strip()
        x, y = int(x), int(y)
        while not self.check_valid_input(
            x, y, dir, chess_board.shape[0], chess_board.shape[1]
        ):
            print(
                "Invalid Move! (x, y) should be within the board and dir should be one of u,r,d,l."
            )
            text = input("Your move (x,y,dir) or input q to quit: ")
            while len(text.split(",")) != 3 and "q" not in text.lower():
                print("Wrong Input Format!")
                text = input("Your move (x,y,dir) or input q to quit: ")
            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)
            x, y, dir = text.split(",")
            x, y, dir = x.strip(), y.strip(), dir.strip()
            x, y = int(x), int(y)
        my_pos = (x, y)
        startTime = time.time()
        z = self.get_legal_actions()
        endTime = time.time()
        howMuchTime = endTime - startTime
        # print(z)
        # print(len(z))
        isEnd, occupied = self.is_game_over(my_pos, dir)
        print(isEnd)
        print(occupied)
        print(self.game_result(occupied))
        print(str(howMuchTime) + " sec")
        # print(self.check_valid_step((x,y), self.dir_map[dir]));
        return my_pos, self.dir_map[dir]

    def check_valid_input(self, x, y, dir, x_max, y_max):
        return 0 <= x < x_max and 0 <= y < y_max and dir in self.dir_map

    def get_legal_actions(self):

        def check_valid_barrier(end_pos, barrier_dir):
            r, c = end_pos
            if self.chess_board[r, c, barrier_dir]:
                return False
            return True

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
                if (check_valid_barrier(cur_pos, barrier_dir)):
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

    def is_game_over(self, next_pos, dir):
        r,c, = next_pos
        chessCopy = deepcopy(self.chess_board)
        chessCopy[r, c, self.dir_map[dir]] = True
        return self.aStar(chessCopy)

    def game_result(self, occupied):
        board_size = len(self.chess_board)
        if board_size*board_size - occupied > occupied:
            return -1
        elif board_size*board_size - occupied == occupied:
            return 0
        else:
            return 1
