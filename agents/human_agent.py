# Human Input agent
from agents.agent import Agent
from store import register_agent
import sys
import time


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
        x = self.get_legal_actions()
        endTime = time.time()
        howMuchTime = endTime - startTime
        print(x)
        print(len(x))
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

    # def check_endgame(self):
    #     """
    #     Check if the game ends and compute the current score of the agents.

    #     Returns
    #     -------
    #     is_endgame : bool
    #         Whether the game ends.
    #     player_1_score : int
    #         The score of player 1.
    #     player_2_score : int
    #         The score of player 2.
    #     """
    #     # Union-Find
    #     father = dict()
    #     for r in range(self.board_size):
    #         for c in range(self.board_size):
    #             father[(r, c)] = (r, c)

    #     def find(pos):
    #         if father[pos] != pos:
    #             father[pos] = find(father[pos])
    #         return father[pos]

    #     def union(pos1, pos2):
    #         father[pos1] = pos2

    #     for r in range(self.board_size):
    #         for c in range(self.board_size):
    #             for dir, move in enumerate(
    #                 self.moves[1:3]
    #             ):  # Only check down and right
    #                 if self.chess_board[r, c, dir + 1]:
    #                     continue
    #                 pos_a = find((r, c))
    #                 pos_b = find((r + move[0], c + move[1]))
    #                 if pos_a != pos_b:
    #                     union(pos_a, pos_b)

    #     for r in range(self.board_size):
    #         for c in range(self.board_size):
    #             find((r, c))
    #     p0_r = find(tuple(self.p0_pos))
    #     p1_r = find(tuple(self.p1_pos))
    #     p0_score = list(father.values()).count(p0_r)
    #     p1_score = list(father.values()).count(p1_r)
    #     if p0_r == p1_r:
    #         return False, p0_score, p1_score
    #     player_win = None
    #     win_blocks = -1
    #     if p0_score > p1_score:
    #         player_win = 0
    #         win_blocks = p0_score
    #     elif p0_score < p1_score:
    #         player_win = 1
    #         win_blocks = p1_score
    #     else:
    #         player_win = -1  # Tie
    #     if player_win >= 0:
    #         # logging.info(
    #         #     f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
    #         # )
    #     else:
    #         # logging.info("Game ends! It is a Tie!")
    #     return True, p0_score, p1_score