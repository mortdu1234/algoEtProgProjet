from MazeMaker import Maze, MazeColor
from tests_MazeMaker_EvaluationsEmpiriques import evaluer_complexite, evaluer_taille_chemin
import random
import os

SAVE_FOLDER = "tests/MazeMaker/"

os.makedirs(SAVE_FOLDER, exist_ok=True)

evaluation_complexite = False
if evaluation_complexite:
    print("="*50+"\nEvaluation Complexité création de Labyrinthe\n"+"="*50)
    figure = evaluer_complexite([i for i in range(5, 500, 30)])
    figure.savefig(SAVE_FOLDER+"ComplexiteCreation.png")

evaluation_chemin = False
if evaluation_chemin:
    print("="*50+"\nEvaluation taille Chemins\n"+"="*50)
    for j in range(1):
        fig = evaluer_taille_chemin([i for i in range(10, 511, 50)], repetitions=10)
        fig.savefig(SAVE_FOLDER+f"longueur_paths{j}.png")


show = False
value = True
for i in [20, 100, 500]:
    print("="*50+f"\nGeneration d'un labyrinthe {i}x{i} avec la seed 20\n"+"="*50)
    maze = Maze(i, 20)
    
    fig_maze = maze.get_fig_pixel_map(title="Pixel Map", show_image=show)
    fig_dijkstra = maze.get_fig_dijkstra_map(title="Dijkstra Map", show_image=show, show_values=value)
    fig_dimention = maze.get_fig_dimention_map(title="Dimention Map", show_image=show, show_values=value)

    fig_maze.savefig(SAVE_FOLDER+f"figure_{i}_maze.png")
    fig_dijkstra.savefig(SAVE_FOLDER+f"figure_{i}_dijkstra.png")
    fig_dimention.savefig(SAVE_FOLDER+f"figure_{i}_dimention.png")



    best_path = maze.get_path()
    fig_path = maze.set_path(best_path, MazeColor.BEST_PATH, MazeColor.BEST_FINISH, "Best Path", show_image=show)
    fig_path.savefig(SAVE_FOLDER+f"figure_{i}_bestPath.png")

    value = False
