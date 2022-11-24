# Enter your code here. Read input from STDIN. Print output to STDOUT
class Node (object):
	def __init__(self,value,point):
		self.value = value
		self.point = point
		self.refresh()

	def refresh(self):
		self.parent = None
		self.H = 0
		self.G = 0

	def move_cost(self,other):
		return 0 if self.value == '.' else 1
		
# def children(point,grid):
# 	x,y = point.point

# 	links = []
# 	# for d in [(max(0, x-1), y),(x,max(0, y - 1)),(x,min(len(grid[0])-1, y + 1)),(min(len(grid)-1, x+1),y)]:
# 	for i in [x-1, x, x+1]:
# 		for j in [y-1, y, y+1]:
# 			if i != x or j != y:
# 				if (i >= 0 and j >= 0 and i < len(grid) and j < len(grid[0])):
# 					links.append(grid[i][j])

# 	ret = [
#         link 
#         for link in links
#             if (link.value != '%')ç]

# 	return ret

def neighbours(self, cur_pos):
	neighbours = []
	r,c = cur_pos
	for dir, move in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
		if self.chess_board[r, c, dir]:
			continue
		a, b = move
		next_pos = (r+a, c+b)
		# print((next_pos, cur_step + 1))
		neighbours.append(((r,c), 1, manhattan(cur_pos, self.end_pos)))
	return neighbours


def manhattan(cur_pos, end_pos):
	cr, cc = cur_pos
	er, ec = end_pos
	return (abs(cr - er) + abs(cc-ec))

def aStar(start, end_pos):
	#The open and closed sets
	pqueue = set()
	closedset = set()
	#Current point is the starting point
	cur_pos = (start, 0, manhattan(cur_pos, end_pos))
	#Add the starting point to the priorityQueue
	pqueue.add(cur_pos)
	#While the open set is not empty
	while pqueue:
		#Find the item in the open set with the lowest G + H score
		cur_pos = min(pqueue, key=lambda o:o.G + o.H)
		# If it is the item we want, retrace the path and return it

		# If goal is reached, then game has not ended and return
		if cur_pos == end_pos: 
			return False
		pqueue.remove(cur_pos)    
		closedset.add(cur_pos)
		#Loop through the node's children/siblings
		for next_pos in neighbours(cur_pos):
			next_coord, next_g, next_h = next_pos
			#If it is already in the closed set, skip it
			if next_pos in closedset:
				continue
			#Otherwise if it is already in the open set
			if next_pos in pqueue:
				#Check if we beat the G score 
				(cr, cc), cur_g, cur_h = cur_pos
				new_g = cur_g + 1
				if next_g > new_g:
					#If so, update the node to have a new parent
					next_g = new_g
					# next_pos.parent = cur_pos
			else:
				#If it isn't in the open set, calculate the G and H score for the node
				next_g = cur_g + 1
				next_h = manhattan(next_pos, end_pos)
				#Set the parent to our current item
				# next_pos.parent = cur_pos
				#Add it to the set
				pqueue.add((next_coord, next_g, next_h))
	#return empty list, as there is not path leading to destination
	return True
