import time
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
import pygame

from .utils import (can_move, dijkstra_sum, get_state, is_deadlock, is_solved,
                    manhattan_sum, print_state)


def astar(matrix, player_pos, widget=None, visualizer=False, heuristic='manhattan'):
	print(f'A* - {heuristic.title()} Heuristic')
	heur = '[A*]' if heuristic == 'manhattan' else '[Dijkstra]'
	shape = matrix.shape
	initial_state = get_state(matrix)
	initial_cost = curr_depth = 0
	if heuristic == 'manhattan':
		curr_cost = manhattan_sum(initial_state, player_pos, shape)
	else:
		distances = defaultdict(lambda: [])
		curr_cost = dijkstra_sum(initial_state, player_pos, shape, distances)
	seen = {None}
	heap = []
	heappush(heap, (initial_cost, curr_cost, initial_state, player_pos, curr_depth, ''))
	moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	direction = {
		(1, 0): 'D',
		(-1, 0): 'U', 
		(0, -1): 'L',
		(0, 1): 'R',
	}
	while heap:
		if widget:
			pygame.event.pump()
		_, curr_cost, state, pos, depth, path = heappop(heap)
		seen.add(state)
		for move in moves:
			new_state, move_cost = can_move(state, shape, pos, move)
			deadlock = is_deadlock(new_state, shape)
			if new_state in seen or deadlock:
				continue
			new_pos = pos[0] + move[0], pos[1] + move[1]
			if heuristic == 'manhattan':
				new_cost = manhattan_sum(new_state, new_pos, shape)
			else:
				new_cost = dijkstra_sum(new_state, new_pos, shape, distances)
			if new_cost == float('inf'):
				continue
			heappush(heap, (
				move_cost + curr_cost,
				new_cost,
				new_state,
				new_pos,
				depth + 1,
				path + direction[move],
			))
			if is_solved(new_state):
				print(f'{heur} Solution found!\n\n{path + direction[move]}\nDepth {depth + 1}\n')
				if widget and visualizer:
					widget.solved = True
					widget.set_text(f'{heur} Solution Found!\n{path + direction[move]}', 20)
					pygame.display.update()
				return (path + direction[move], depth + 1)
			if widget and visualizer:
				widget.set_text(f'{heur} Solution Depth: {depth + 1}\n{path + direction[move]}', 20)
				pygame.display.update()
	print(f'{heur} Solution not found!\n')
	if widget and visualizer:
		widget.set_text(f'{heur} Solution Not Found!\nDepth {depth + 1}', 20)
		pygame.display.update()
	return (None, -1 if not heap else depth + 1)


def solve_astar(puzzle, widget=None, visualizer=False, heuristic='manhattan'):
	matrix = puzzle
	where = np.where((matrix == '*') | (matrix == '%'))
	player_pos = where[0][0], where[1][0]
	return astar(matrix, player_pos, widget, visualizer, heuristic)

	
if __name__ == '__main__':
	start = time.time()
	solve_astar(np.loadtxt('levels/lvl5.dat', dtype='<U1'), heuristic='dijkstra')
	print(f'Runtime: {time.time() - start} seconds')
