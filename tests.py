
from Zmaze_maker2 import Maze

m = Maze(20, 20)
while True:
    x = int(input("x="))
    y = int(input("y="))
    map, dijkstra, dimention = m.get_pixels(x, y)
    print(f"map: {map}, dijkstra: {dijkstra}, dimention: {dimention}")