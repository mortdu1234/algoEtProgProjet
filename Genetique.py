from MazeMaker import Maze, MazeColor
from random import randint as rdm   
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from Logs import Logs
os.makedirs("logs/explorations", exist_ok=True)

LOGGER = Logs()

import matplotlib.pyplot as plt


def fusionner_figures_horizontale(fig1, fig2):
    """
    Fusionne deux figures matplotlib horizontalement
    avec tailles cohérentes et légendes correctes.
    """
    new_fig, axes = plt.subplots(
        1, 2,
        figsize=(
            fig1.get_figwidth() + fig2.get_figwidth(),
            max(fig1.get_figheight(), fig2.get_figheight())
        )
    )

    for src_fig, target_ax in zip([fig1, fig2], axes):
        src_ax = src_fig.axes[0]  # hypothèse : 1 axe par figure

        # Lignes
        for line in src_ax.get_lines():
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker()
            )

        # Images (imshow)
        for img in src_ax.images:
            target_ax.imshow(
                img.get_array(),
                extent=img.get_extent(),
                origin=img.origin,
                cmap=img.get_cmap(),
                alpha=img.get_alpha(),
                aspect = src_ax.get_aspect()

            )

        # Propriétés
        target_ax.set_xlim(src_ax.get_xlim())
        target_ax.set_ylim(src_ax.get_ylim())
        target_ax.set_title(src_ax.get_title())
        target_ax.set_xlabel(src_ax.get_xlabel())
        target_ax.set_ylabel(src_ax.get_ylabel())

        # Légende CORRECTE
        handles, labels = src_ax.get_legend_handles_labels()
        if labels:
            target_ax.legend(handles, labels)

    plt.tight_layout()
    return new_fig

class Population:
    """Gestion de la population
    """
    LONGUEURS_MAX: int = 5 # longueurs max des chemins des individues
    TAUX_MUTATION: float = 0.2 # taux de mutation dans les chemins des individues
    TAUX_GARDEE: float = 0.3 # taux des meilleurs individues gardé 
    NOMBRE_INDIVIDUES: int = 100 # nombre total d'individues
    NOMBRE_GENERATION: int = 10 # nombre de génération d'individues
    
    @staticmethod
    def set_longueurs_max(new_longueurs: int):
        """Définie la longueur max des chemins de chaques individues

        Args:
            new_longueurs (int): longueur des chemins
        """
        # assert isinstance(new_longueurs, int) and new_longueurs > 0
        Population.LONGUEURS_MAX = new_longueurs
    
    @staticmethod
    def set_mutation(new_mutation: float):
        """Définie le taux de mutation des individues

        Args:
            new_mutation (float): coefficient de mutation
        """
        # assert isinstance(new_mutation, float) and 0 <= new_mutation <= 1
        Population.TAUX_MUTATION = new_mutation
    
    @staticmethod
    def set_gardee(new_gardee: float):
        """Définie le taux d'Individues guardée lors de la Sélection

        Args:
            new_gardee (float): coefficient de gardée
        """
        # assert isinstance(new_gardee, float) and 0 <= new_gardee <= 1
        Population.TAUX_GARDEE = new_gardee
    
    @staticmethod
    def set_individues(nb_individues: int):
        """Définie le nombre d'individues dans la population

        Args:
            nb_individues (int): nombre d'individues
        """
        # assert isinstance(nb_individues, int) and nb_individues > 1
        Population.NOMBRE_INDIVIDUES = nb_individues

    @staticmethod
    def set_generation(nb_generation: int):
        """Définie le nombre de générations maximales

        Args:
            nb_generation (int): nombre de générations maximales
        """
        # assert isinstance(nb_generation, int) and nb_generation > 0
        Population.NOMBRE_GENERATION = nb_generation

    def affichage_barre_progression(self, current: int):
        """affiche la barre de progression

        Args:
            current (int): génération actuelle
        """
        longueur_barre = 50
        rempli = int(longueur_barre * current / Population.NOMBRE_GENERATION)
        barre = "█" * rempli + "─" * (longueur_barre - rempli)
        pourcent = (current / Population.NOMBRE_GENERATION) * 100
        ligne = f"[{barre}] {pourcent:5.1f}% ({current:3d}/{Population.NOMBRE_GENERATION})"
        print(f"\r{ligne}", end="", flush=True)
    
    
    def __init__(self, maze: 'Maze'):
        # Population initiale
        self.individues: list[Individue] = []
        self.maze: 'Maze' = maze
        self.scores_min: list[int] = []
        self.scores_avg: list[float] = []
        self.scores_max: list[int] = []
        self.distances: list[int] = []

        for _ in range(Population.NOMBRE_INDIVIDUES):
            individue: Individue = Individue([rdm(0, 7) for _ in range(Population.LONGUEURS_MAX)], maze.start_coords)
            self.individues.append(individue)


    def __str__(self):
        res = ""
        for idv in self.individues:
            res += f"{idv}\n"
        return res

    def note_statistiques(self):
        scores = [indiv.score for indiv in self.individues]
        self.scores_min.append(min(scores)) # type: ignore
        self.scores_avg.append(sum(scores) / len(scores)) # type: ignore
        self.scores_max.append(max(scores)) # type: ignore
        coords_final = self.individues[0].get_path(self.maze)[-1]
        self.distances.append(self.maze.dijkstra_map[coords_final[0]][coords_final[1]])

    def simulation(self, picture_each_x_generation=10):

        self.calcul_fitness()
        self.tri_individue()
        print("Début de la simulation génétique...")
        s2 = time.perf_counter()
        cpt = 0
        for i in range(Population.NOMBRE_GENERATION):
            self.note_statistiques()

            self.selection()
            # assert len(self.individues) == (Population.NOMBRE_INDIVIDUES*Population.TAUX_GARDEE), f"{len(self.individues)} il manque du monde"
            
            
            self.reproduction()
            # assert len(self.individues) == (Population.NOMBRE_INDIVIDUES), f"{len(self.individues)} il manque du monde"
            
            
            self.mutation()
            # assert len(self.individues) == (Population.NOMBRE_INDIVIDUES), f"{len(self.individues)} il manque du monde"
            
            self.add_explorations()
            
            self.calcul_fitness()
            # assert len(self.individues) == (Population.NOMBRE_INDIVIDUES), f"{len(self.individues)} il manque du monde"
       

            self.tri_individue()
            self.affichage_barre_progression(i+1)
            
            cpt += 1
            nb_generation = picture_each_x_generation
            if cpt % nb_generation == 0:
                fig1 = self.maze.get_fig_explored_phase_map(title=f"Explored Map {i-nb_generation+1}-{i}", show_image=False, show_values=True)
                fig1.savefig(f"logs/explorations/{i-nb_generation+1}-{i}_explored.png")
            
                fig2 = self.maze.set_path(self.individues[0].get_path(self.maze), MazeColor.BEST_PATH, MazeColor.BEST_FINISH)
                fig2.savefig(f"logs/explorations/{i-nb_generation+1}-{i}_best_path.png")  

                fig3 = fusionner_figures_horizontale(fig1, fig2)
                fig3.savefig(f"logs/explorations/{i-nb_generation+1}-{i}.png")  

                LOGGER.write_generation(self.maze, self.individues[0], i, i-nb_generation+1)


        e2 = time.perf_counter()
        print(f"\nTemps total de la simulation : {e2 - s2:.0f}s")
        print("\nSimulation terminée.")
        return self.scores_min, self.scores_max, self.scores_avg, self.distances

    def calcul_fitness(self):
        # assert len(self.individues) == Population.NOMBRE_INDIVIDUES, 'il manque du monde'
        for individu in self.individues:
            individu.fitness(self.maze)
            individu.place_pheromone2(self.maze)

    def tri_individue(self):
        """effectue un tri des individues de celui qui as le score le plus faible a celui qui a le score le plus grand
        """
        self.individues.sort(key=lambda ind: ind.score) # type: ignore

    def selection(self):
        nombre: int = round(Population.NOMBRE_INDIVIDUES * Population.TAUX_GARDEE)
        self.individues = self.individues[:nombre]

    def reproduction(self):
        n = len(self.individues)
        enfants = []
        for _ in range(n, Population.NOMBRE_INDIVIDUES):
            parent1 = random.choice(self.individues)
            parent2 = random.choice(self.individues)
            enfants.append(Individue.fusion(parent1, parent2))
        self.individues.extend(enfants)
        
    def add_explorations(self):
        for indiv in self.individues:
            path = indiv.get_path(self.maze)
            for x, y in path:
                self.maze.add_exploration(x, y)
    
    def mutation(self):
        nb_mutant = round(Population.NOMBRE_INDIVIDUES * Population.TAUX_GARDEE * Population.TAUX_MUTATION) # nombre de mutant
        nb_mutation = 1 # nombre de mutation par mutant
        idx_mutants = []

        for _ in range(nb_mutant):
            # selection du mutant
            idx_mutant = random.randint(round(Population.NOMBRE_INDIVIDUES * Population.TAUX_GARDEE), Population.NOMBRE_INDIVIDUES-1)
            while idx_mutant in idx_mutants:
                idx_mutant = random.randint(round(Population.NOMBRE_INDIVIDUES * Population.TAUX_GARDEE), Population.NOMBRE_INDIVIDUES-1)
                # print(idx_mutant, idx_mutants)
            idx_mutants.append(idx_mutant)
            mutant : Individue = self.individues[idx_mutant]
            # print(f"modification de l'individue {mutant}")

            # selection des mutations du mutants
            idx_mutations = []
            for _ in range(nb_mutation):
                idx_mutation = random.randint(0, mutant.length()-1)
                while idx_mutation in idx_mutations:
                    idx_mutation = random.randint(0, mutant.length()-1)
                idx_mutations.append(idx_mutation)

                new_value = random.randint(0, 7)
                # print(f"\tidx:{idx_mutation} | {mutant.mouvements[idx_mutation]} -> {new_value}")
                mutant.mouvements[idx_mutation] = new_value
            # print(f"individue modifié           {mutant}")
            





class Individue:
    TEMPLATE_MOVE = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)] # liste des mouvements possibles
    CUT_OFFSET = 0.05 # % d'offset du cut autour du milieux
    PENALITES = {
            "sortie de terrain": 1,
            "foncer dans un mur": 1,
            "bonus d'arrivee": 0,
            "position final": 3,
            "retour en arriere": 1,
        }
    @staticmethod
    def set_offset(offset: float):
        # assert isinstance(offset, float) and 0 <= offset <= 1
        Individue.CUT_OFFSET = offset
    @staticmethod
    def set_penalite(maze: 'Maze'):
        Individue.PENALITES = {
            "sortie de terrain": 1,
            "foncer dans un mur": 1,
            # "bonus d'arrivee": -maze.get_pixels(maze.start_coords[0], maze.start_coords[1])[1],
            "bonus d'arrivee": 0,
            "position final": 3,
            "retour en arriere": 1,
        }


    def __init__(self, mouvements_effectues: 'list[int]', start_point: 'tuple[int, int]'):
        """gestion des individues

        Args:
            mouvements_effectues (list[int]): liste des mouvements effectuer par l'individue
            start_point (tuple[int, int]): point de départ de l'individue
        """
        #assert Individue.PENALITES is not None, "il faut configurer les pénalités"
        self.mouvements: 'list[int]' = mouvements_effectues
        self.start_point: tuple[int, int] = start_point
        self.score: int | None = None
        self.penalities = Individue.PENALITES
        self.counter = {key:0 for key in Individue.PENALITES}
        self.path = None




    def length(self) -> int:
        """renvoie la longueur du chemin de l'individue

        Returns:
            int: longueur du chemin
        """
        return len(self.mouvements)
    
    def __str__(self):
        return f"{self.mouvements} | score : {self.score}"
    
    def get_path(self, maze: 'Maze') -> 'list[tuple[int, int]]':
        """récupère la liste des case visité par l'individue

        Returns:
            list[tuple[int, int]]: liste des coordonnées visités
        """
        if self.path is not None:
            return self.path
        path = [self.start_point]
        for move in self.mouvements:
            last_x, last_y = path[-1]
            add = Individue.TEMPLATE_MOVE[move]
            next_x, next_y = last_x+add[0], last_y+add[1]
            # print(f"mouvement {move} from {last_x} {last_y} to {next_x} {next_y}", end=" ")

            if not(0 <= next_x < maze.size) or not(0 <= next_y < maze.size):
                # print(f"\nOutOfBand {next_x} {next_y}")
                next_x = last_x
                next_y = last_y
            # print(f"valueNext {maze.get_pixels(next_x, next_y)[0]}")
            
            if maze.pixels_map[next_x][next_y] == MazeColor.WALL:
                # print(f"Wall {next_x} {next_y}")
                next_x = last_x
                next_y = last_y



            
            path.append((next_x, next_y))
        self.path = path
        return path
    
    def place_pheromone(self, maze: 'Maze'):
        """place des feromones sur le chemin de l'individue

        Args:
            maze (Maze): labyrinthe dans lequel l'individue se déplace
        """
        path = self.get_path(maze)

        idx = 0
        while idx < len(path):
            x, y = path[idx]
            # calcul du nombre de chemin possible
            voisins = maze.get_voisins(x, y)
            nb_chemin_possible = 0
            for vx, vy in voisins:
                if maze.pixels_map[vx][vy] != MazeColor.WALL:
                    nb_chemin_possible += 1
            
            # on place un phéromone
            if nb_chemin_possible == 1 and maze.pixels_map[x][y] != MazeColor.WALL and (x, y) != maze.start_coords and (x, y) != maze.end_coords:
                maze.set_wall(x, y)
                idx -= 1
            else:
                idx += 1
    
    def place_pheromone2(self, maze: 'Maze'):
        path = self.get_path(maze)
        idx = 0
        size = maze.size
        pixels = maze.pixels_map
        start = maze.start_coords
        end = maze.end_coords

        while 0 <= idx < len(path):
            x, y = path[idx]

            if (x, y) == start or (x, y) == end:
                idx += 1
                continue

            if pixels[x][y] == MazeColor.WALL:
                idx += 1
                continue

            nb_chemin_possible = 0

            # voisins codés en dur (beaucoup plus rapide)
            for dx, dy in Individue.TEMPLATE_MOVE:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if pixels[nx][ny] != MazeColor.WALL:
                        nb_chemin_possible += 1
                        if nb_chemin_possible > 1:
                            break

            if nb_chemin_possible == 1:
                maze.set_wall(x, y)
                idx -= 1   # on revient en arrière
            else:
                idx += 1

    

    @staticmethod
    def fusion(individue1: 'Individue', individue2: 'Individue'):
        longueur1 = len(individue1.mouvements)
        offset1 = round(longueur1*Individue.CUT_OFFSET)
        milieu1 = longueur1//2
        pos_cut1 = random.randint(milieu1-offset1, milieu1+offset1)

        longueur2 = len(individue2.mouvements)
        offset2 = round(longueur2*Individue.CUT_OFFSET)
        milieu2 = longueur2//2
        pos_cut2 = random.randint(milieu2-offset2, milieu2+offset2)

        new_movement = individue1.mouvements[:pos_cut1] + individue2.mouvements[pos_cut2:]

        if len(new_movement) > Population.LONGUEURS_MAX:
            new_movement = new_movement[:Population.LONGUEURS_MAX]
        
        # assert len(new_movement) <= Population.LONGUEURS_MAX, "la longueur de l'individue est trop grande après fusion"
        return Individue(new_movement, individue1.start_point)
    


    def fitness(self, maze: 'Maze') -> int:
        """Calcule le score de fitness de l'individu"""
        x, y = self.start_point
        self.counter = {key:0 for key in Individue.PENALITES}
        
        visited = [(x, y)]
        for step in self.mouvements:
            # print(f"move : {step} => {Individue.TEMPLATE_MOVE[step]}")
            last_x, last_y = visited[-1]
            add = Individue.TEMPLATE_MOVE[step]
            next_x, next_y = last_x+add[0], last_y+add[1]

            if not(0 <= next_x < maze.size) or not(0 <= next_y < maze.size):
                next_x = last_x
                next_y = last_y
                self.counter["sortie de terrain"] += 1
            # print(f"valueNext {maze.get_pixels(next_x, next_y)[0]}")
            
            if maze.pixels_map[next_x][next_y] == MazeColor.WALL:
                # print(f"Wall {next_x} {next_y}")
                next_x = last_x
                next_y = last_y
                self.counter["foncer dans un mur"] += 1

            if (last_x, last_y) == maze.end_coords:
                self.counter["bonus d'arrivee"] = 1
            
            if (next_x, next_y) in visited and (next_x, next_y) != (last_x, last_y):
                self.counter["retour en arriere"] += 1


            last_x, last_y = next_x, next_y
            visited.append((last_x, last_y))
        
        self.path = visited
        # assert visited == self.get_path(maze), f"il y a un problemes de path\n{visited}\n{self.get_path(maze)}"
    

        explication = ""
        self.counter["position final"] = maze.dijkstra_map[last_x][last_y]
        # assert self.counter["position final"] != -1, f"probleme de coordonnée dijkstra, {self.counter['position final']} valeur négative {last_x} {last_y}\n{self.get_path(maze)}\n{visited}"
        
        self.score = 0
        
        key = "position final"
        self.score += self.counter[key] ** Individue.PENALITES[key]
        explication += f"{self.counter[key]} ** {Individue.PENALITES[key]} + "

        key = "sortie de terrain"
        self.score += Individue.PENALITES[key] * self.counter[key]
        explication += f"{Individue.PENALITES[key]} * {self.counter[key]} + "
        
        key = "foncer dans un mur"
        self.score += Individue.PENALITES[key] * self.counter[key]
        explication += f"{Individue.PENALITES[key]} * {self.counter[key]} + "
        
        key = "bonus d'arrivee"
        self.score += Individue.PENALITES[key] * self.counter[key]
        explication += f"{Individue.PENALITES[key]} * {self.counter[key]} + "
        
        key = "retour en arriere"
        self.score += Individue.PENALITES[key] * self.counter[key]
        explication += f"{Individue.PENALITES[key]} * {self.counter[key]} "
        

        explication += f"= {self.score}"
        self.scoreExplication = explication
        # LOGGER.write(f"{visited} => {last_x} {last_y} : {maze.get_pixels(last_x, last_y)[1]} | {explication}")
        
        
        return self.score



    
            
def get_chemin_robot(path):
    """fonction qui convertis le chemin envoyé par le labyrinthe en chemin pour les robot

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = []
    # print(path)
    for idx in range(1, len(path)):
        x1, y1 = path[idx-1]
        x2, y2 = path[idx]

        res.append(Individue.TEMPLATE_MOVE.index((x2-x1, y2-y1)))
    return res

def merge_figures(fig1, fig2, titles=None):
    """
    Fusionne deux figures matplotlib en une seule image
    
    Args:
        fig1: Première figure
        fig2: Deuxième figure
        titles: Tuple de titres optionnels (titre1, titre2)
    
    Returns:
        Figure combinée
    """
    # Créer la figure combinée
    fig_combined = plt.figure(figsize=(16, 7))
    
    # Convertir les figures en arrays numpy
    fig1.canvas.draw()
    fig2.canvas.draw()
    
    img1 = np.array(fig1.canvas.renderer.buffer_rgba())
    img2 = np.array(fig2.canvas.renderer.buffer_rgba())
    
    # Créer les sous-graphiques
    ax1 = fig_combined.add_subplot(1, 2, 1)
    ax2 = fig_combined.add_subplot(1, 2, 2)
    
    # Afficher les images
    ax1.imshow(img1)
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.axis('off')
    
    # Ajouter les titres si fournis
    if titles:
        ax1.set_title(titles[0], fontsize=14, pad=10)
        ax2.set_title(titles[1], fontsize=14, pad=10)
    
    plt.tight_layout()
    
    return fig_combined





