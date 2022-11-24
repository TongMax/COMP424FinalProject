def check_valid_step(self, end_pos):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        end_pos : tuple
            The end position of the agent.
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
                print("This is dir: ",dir)
                print("This is move: ", move)
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
                # print((next_pos, cur_step + 1))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached


    
    def get_legal_actions(self): 
        legal_actions_queue = {}
        # print("Max step is: ", self.max_step)
        for i in range(self.max_step+1):
            # print("Value of i is: ", i)
            for j in range(i+1):
                # print("Value of j is: ", i)
                r,c = self.start_pos
                mr = j
                mc = i-j
                extraCol = 0
                extraRow = 0
                if (r-mr < 0):
                    temp = mr-r
                    extraCol = c+mc+temp
                if (c-mc < 0):
                    temp = mc-c
                    extraRow = r+mr+temp
                next_pos = [(r+mr,c+mc), (extraRow,c+mc), (r+mr,extraCol), (extraRow,extraCol)]
                # print(next_pos)
                for cur_pos in next_pos:
                    # print(x)
                    if (self.check_valid_step(cur_pos)):
                        # Remember to change map to 0,1,2,3 for the 4 possible cardinal directions of the cell
                        for barrier_dir in range(4):
                            if (self.check_valid_barrier(cur_pos, barrier_dir)):
                                legal_actions_queue[(cur_pos, barrier_dir)] = barrier_dir
        return legal_actions_queue




# Code for A2.py # # # # # # # # # #
# def next_move(pacman,food,grid):
# 	#Convert all the points to instances of Node
# 	for x in range(len(grid)):
# 		for y in range(len(grid[x])):
# 			grid[x][y] = Node(grid[x][y],(x,y))
# 	#Get the path
# 	path = aStar(grid[pacman[0]][pacman[1]],grid[food[0]][food[1]],grid)
# 	path = aStar(grid[pacman[0]][pacman[1]],grid[food[0]][food[1]],grid)
# 	path = aStar(grid[pacman[0]][pacman[1]],grid[food[0]][food[1]],grid)
# 	path = aStar(grid[pacman[0]][pacman[1]],grid[food[0]][food[1]],grid)
# 	#Output the path
# 	print (len(path) - 1)
# 	for node in path:
# 		x, y = node.point
# 		print (x, y)

# pacman_x, pacman_y = [ int(i) for i in input().strip().split() ]
# food_x, food_y = [ int(i) for i in input().strip().split() ]
# x,y = [ int(i) for i in input().strip().split() ]
 
# grid = []
# for i in range(0, x):
# 	grid.append(list(input().strip()))
 
# next_move((pacman_x, pacman_y),(food_x, food_y), grid)
            
		# 	path = []
		# 	while cur_pos.parent:
		# 		path.append(cur_pos)
		# 		cur_pos = cur_pos.parent
		# 	path.append(cur_pos)
		# 	return path[::-1]
		# Remove the item from the open set
		
		#Add it to the closed set