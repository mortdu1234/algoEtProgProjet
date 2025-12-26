from Zmaze_maker2 import Maze, MazeColor
import random
import os


os.makedirs("tests", exist_ok=True)
os.makedirs("tests/maze_maker", exist_ok=True)
show = False
value = True

maze1 = Maze(20, 20)
map, dijkstra, dimention = maze1.get_pixels(18, 14)
print(f"map: {map}, dijkstra: {dijkstra}, dimention: {dimention}")
for line in maze1.dijkstra_map:
    print(line)





print(f"random value before {random.random()}")
maze2 = Maze(20, 20)
print(f"random value after {random.random()}")
print(f"random value after {random.random()}")
fig_maze = maze2.get_fig_pixel_map(title="", show_image=show)
fig_dijkstra = maze2.get_fig_dijkstra_map(title="", show_image=show, show_values=value)
fig_dimention = maze2.get_fig_dimention_map(title="", show_image=show, show_values=value)


fig_maze.savefig("tests/maze_maker/figure_maze.png")
fig_dijkstra.savefig("tests/maze_maker/figure_dijkstra.png")
fig_dimention.savefig("tests/maze_maker/figure_dimention.png")

fig_path=maze2.set_path([(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (1, 9)], MazeColor.BEST_PATH, title="", show_image=show)
fig_path.savefig("tests/maze_maker/figure_path.png")

