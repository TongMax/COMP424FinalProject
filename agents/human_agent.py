# Human Input agent
from agents.agent import Agent
from store import register_agent
import sys


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
        print(self.get_legal_actions())
        # print(self.check_valid_step((x,y), self.dir_map[dir]));
        return my_pos, self.dir_map[dir]

    def check_valid_input(self, x, y, dir, x_max, y_max):
        return 0 <= x < x_max and 0 <= y < y_max and dir in self.dir_map

    def check_valid_step(self, end_pos):
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
        r,c = end_pos
        if (r < 0 or c < 0):
            return False
        if (self.start_pos==end_pos):
            return True

        # BFS
        state_queue = [(self.start_pos, 0)]
        visited = {tuple(self.start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            # print(cur_step)
            # print("This is ", cur_pos)
            r, c = tuple(cur_pos)
            
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                if self.chess_board[r, c, dir]:
                    continue
                a, b = move
                next_pos = (r+a, c+b)
                if (next_pos==self.adv_pos) or tuple(next_pos) in visited:
                    continue
                if (next_pos==end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                print((next_pos, cur_step + 1))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_valid_barrier(self, end_pos, barrier_dir):
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        return True
    
    def get_legal_actions(self): 
        legal_actions_queue = []
        print(self.max_step)
        for i in range(self.max_step):
            for j in range(i):
                r,c = self.start_pos
                mr = j
                mc = i-j
                next_pos = [(r+mr,c+mc), (r-mr,c+mc), (r+mr,c-mc), (r-mr,c-mc)]
                print(next_pos)
                for x in next_pos:
                    if (self.check_valid_step(next_pos)):
                        # Remember to change map to 0,1,2,3 for the 4 possible cardinal directions of the cell
                        for k in range(3):
                            if (self.check_valid_barrier(next_pos, k)):
                                legal_actions_queue.append((next_pos, k))
        return legal_actions_queue