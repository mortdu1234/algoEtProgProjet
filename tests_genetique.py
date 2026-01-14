from Genetique import Population, Individue
from MazeMaker import Maze, MazeColor
import os
import cProfile
import pstats

import matplotlib.pyplot as plt

def plot_3_curves_save(y1, y2, y3, filename, labels=("Min", "Moyenne", "Max")):
    plt.figure(figsize=(10, 6))
    plt.plot(y1, label=labels[0])
    plt.plot(y2, label=labels[1])
    plt.plot(y3, label=labels[2])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def courbe_loss(values, filename, labels="loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(values, label=labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def clear_exploration():
    path = "logs/explorations"
    files = os.listdir(path)
    for file in files:
        os.remove(f"{path}/{file}")

def main():
    clear_exploration()
    os.makedirs("tests", exist_ok=True)
    os.makedirs("tests/genetique", exist_ok=True)
    os.makedirs("tests/genetique/param", exist_ok=True)
    show = False
    value = True
    maze = Maze(50)
    


    fig = maze.get_fig_pixel_map(title="", show_image=show)
    fig.savefig("tests/genetique/init.png")

    # parametres
    Population.set_gardee(0.02)
    Population.set_generation(1000)
    Population.set_individues(500)
    Population.set_longueurs_max(500)
    Population.set_mutation(0.9)
    Individue.set_offset(0.10)

    population = Population(maze)

    score_min, score_max, score_avg , dist = population.simulation(picture_each_x_generation=10)
    plot_3_curves_save(score_min, score_max, score_avg, "tests/genetique/param/c1.png")
    print(dist)
    courbe_loss(score_min, "tests/genetique/param/loss.png")


    fig = maze.get_fig_explored_full_map(title="", show_values=value, show_image=show)
    fig.savefig("tests/genetique/exploredFull.png")
    fig = maze.get_fig_pixel_map(title="", show_image=show)
    fig.savefig("tests/genetique/final.png")




if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime").print_stats(40)


    
    