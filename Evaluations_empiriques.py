import time
import matplotlib.pyplot as plt
from Maze_maker import Maze


def evaluer_complexite(valeurs_n, repetitions=3, log_scale=False):
    """
    Compare la complexité empirique de deux fonctions.
    
    Paramètres :
    ------------
    valeurs_n : liste des valeurs de n à tester.
    repetitions : nombre de répétitions pour lisser les mesures.
    """
    temps_f1 = []
    temps_f2 = []
    
    # Variables

    for n in valeurs_n:
        print(f"n={n}")
        # Variables
        maze = Maze(n)
        # Mesure du temps pour f1
        start = time.perf_counter()
        for _ in range(repetitions):
            maze.set_up_maze()
        end = time.perf_counter()
        temps_f1.append((end - start) / repetitions)
        
        # Mesure du temps pour f2

    
    # --- Graphique ---
    fig = plt.figure(figsize=(8, 5))
    plt.plot(valeurs_n, temps_f1, marker='o', label="Construction Maze")
    plt.xlabel("Taille d'entrée (n)")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.title("Comparaison empirique de la complexité")
    plt.legend()
    
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    
    if log_scale:
        plt.yscale("log", base=10)
        plt.title("Comparaison empirique de la complexité (échelle logarithmique)")
    
    return fig


def evaluer_taille_chemin(valeurs_n, repetitions=3, log_scale=False):
    """
    Compare la complexité empirique de deux fonctions.
    
    Paramètres :
    ------------
    valeurs_n : liste des valeurs de n à tester.
    repetitions : nombre de répétitions pour lisser les mesures.
    """
    longueurs_f1 = []
    temps_f2 = []
    
    # Variables

    for n in valeurs_n:
        print(f"n={n}", end=" ", flush=True)

        # Variables


        # Mesure du temps pour f1
        longueurs = []
        for i in range(repetitions):    
            maze = Maze(n)
            maze.set_up_maze()
            maze.apply_dijkstra()
            path = maze.set_path()
            longueurs.append(len(path))
            print(f"{i} ", end="", flush=True)

        result = sum(longueurs) / repetitions
        longueurs_f1.append(result)
        print(f"fini -> taille moyenne {result}")
        
        # Mesure du temps pour f2

    
    # --- Graphique ---
    fig = plt.figure(figsize=(8, 5))
    plt.plot(valeurs_n, longueurs_f1, marker='o', label="taille du chemin")
    plt.xlabel("Taille d'entrée (n)")
    plt.ylabel("taille du chemin")
    plt.title("Comparaison empirique de la taille du chemin")
    plt.legend()
    
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    
    if log_scale:
        plt.yscale("log", base=10)
        plt.title("Comparaison empirique de la complexité (échelle logarithmique)")
    
    return fig

if __name__ == "__main__":
    # figure = evaluer_complexite([i for i in range(5, 500, 30)])
    # figure.savefig("images/complexite.png")
    

    

    for i in range(0):
        fig = evaluer_taille_chemin([i for i in range(10, 511, 50)], repetitions=10)
        fig.savefig(f"images/longueur_paths{i}.png")


    for i in [5, 50, 500]:
        maze = Maze(i)
        maze.set_up_maze()
        maze.apply_dijkstra()
        fig = maze.show_dijkstra_map()
        fig.savefig(f"images/tests/lab{i}.png")


    plt.show(block=True)