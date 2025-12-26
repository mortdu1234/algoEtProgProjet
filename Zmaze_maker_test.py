from Zmaze_maker2 import Maze
import random
import os

maze1 = Maze(20)
map, dijkstra, dimention = maze1.get_pixels(0, 0)
print(f"map: {map}, dijkstra: {dijkstra}, dimention: {dimention}")

print(f"random value before {random.random()}")
maze2 = Maze(20, 20)
print(f"random value after {random.random()}")
print(f"random value after {random.random()}")
fig_maze, fig_dijkstra, fig_dimention = maze2.get_fig_of_maze()


os.makedirs("tests", exist_ok=True)
os.makedirs("tests/maze_maker", exist_ok=True)
fig_maze.savefig("tests/maze_maker/figure_maze.png")
fig_dijkstra.savefig("tests/maze_maker/figure_dijkstra.png")
fig_dimention.savefig("tests/maze_maker/figure_dimention.png")
