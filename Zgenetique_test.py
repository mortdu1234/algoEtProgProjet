from Zgenetique2 import Population, Individue
from Zmaze_maker2 import Maze

if __name__ == "__main__":
    maze = Maze(20, 20)

    fig_maze, _, _ = maze.get_fig_of_maze()
    fig_maze.savefig("logs/images/init")

    # parametres
    Population.set_gardee(0.1)
    Population.set_generation(100)
    Population.set_individues(100)
    Population.set_longueurs_max(150)
    Population.set_mutation(0.5)
    Individue.set_offset(0.05)

    population = Population(maze)
    population.simulation()

    fig = maze.get_fig_explored_phase_map(title="", show_values=True, show_image=True)

    fig.savefig("logs/images/init")