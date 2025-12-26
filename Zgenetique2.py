from matplotlib.figure import Figure
from Zmaze_maker2 import Maze, MazeColor
from random import randint as rdm   
import random
import matplotlib.pyplot as plt
import time
import numpy as np


from logs import Logs

LOGGER = Logs()

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
        assert isinstance(new_longueurs, int) and new_longueurs > 0
        Population.LONGUEURS_MAX = new_longueurs
    
    @staticmethod
    def set_mutation(new_mutation: float):
        """Définie le taux de mutation des individues

        Args:
            new_mutation (float): coefficient de mutation
        """
        assert isinstance(new_mutation, float) and 0 <= new_mutation <= 1
        Population.TAUX_MUTATION = new_mutation
    
    @staticmethod
    def set_gardee(new_gardee: float):
        """Définie le taux d'Individues guardée lors de la Sélection

        Args:
            new_gardee (float): coefficient de gardée
        """
        assert isinstance(new_gardee, float) and 0 <= new_gardee <= 1
        Population.TAUX_GARDEE = new_gardee
    
    @staticmethod
    def set_individues(nb_individues: int):
        """Définie le nombre d'individues dans la population

        Args:
            nb_individues (int): nombre d'individues
        """
        assert isinstance(nb_individues, int) and nb_individues > 1
        Population.NOMBRE_INDIVIDUES = nb_individues

    @staticmethod
    def set_generation(nb_generation: int):
        """Définie le nombre de générations maximales

        Args:
            nb_generation (int): nombre de générations maximales
        """
        assert isinstance(nb_generation, int) and nb_generation > 0
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

    def simulation(self):
        LOGGER.log_new("test")

        self.calcul_fitness()
        self.tri_individue()

        print("Début de la simulation génétique...")
        s2 = time.perf_counter()
        cpt = 0
        for i in range(Population.NOMBRE_GENERATION):
            self.note_statistiques()
            self.selection()
            self.reproduction()
            self.mutation()
            self.calcul_fitness()
            self.tri_individue()
            self.affichage_barre_progression(i+1)
            
            cpt += 1
            nb_generation = 10
            if cpt % nb_generation == 0:  
                fig1 = self.maze.get_fig_exploration_map(f"Explored Map {i-nb_generation+1}-{i}")[0]
                fig2 = self.maze.set_path(self.individues[0].get_path(self.maze), MazeColor.BEST_PATH)
                

                fig_combined = merge_figures(
                    fig1, 
                    fig2, 
                    titles=(f"Explored Map {i}", "Best Path")
                )

                fig_combined.savefig(f'logs/exploration/maze_comparison_{i-nb_generation+1}_{i}.png', dpi=300, bbox_inches='tight')
                



        e2 = time.perf_counter()
        print(f"\nTemps total de la simulation : {e2 - s2:.0f}s")

        LOGGER.log_summary_stats(self.scores_min, self.scores_avg, self.scores_max)
        LOGGER.log_plot_statistics(self.scores_min, self.scores_avg, self.scores_max)
        LOGGER.log_plot_convergence_rate(self.scores_min)
        LOGGER.log_plot_loss(self.scores_min, self.scores_avg, self.scores_max)    
        LOGGER.log_save()

        print("\nSimulation terminée.")
        
        
        
            

    def calcul_fitness(self):
        for individu in self.individues:
            if individu.score is None:
                individu.fitness(self.maze)

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
        
    
    def mutation(self):
        nb_mutant = round(Population.NOMBRE_INDIVIDUES * Population.TAUX_MUTATION) # nombre de mutant
        nb_mutation = 1 # nombre de mutation par mutant
        idx_mutants = []

        for _ in range(nb_mutant):
            # selection du mutant
            idx_mutant = random.randint(0, Population.NOMBRE_INDIVIDUES-1)
            while idx_mutant in idx_mutants:
                idx_mutant = random.randint(0, Population.NOMBRE_INDIVIDUES-1)
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
    @staticmethod
    def set_offset(offset: float):
        assert isinstance(offset, float) and 0 <= offset <= 1
        Individue.CUT_OFFSET = offset


    def __init__(self, mouvements_effectues: 'list[int]', start_point: 'tuple[int, int]'):
        """gestion des individues

        Args:
            mouvements_effectues (list[int]): liste des mouvements effectuer par l'individue
            start_point (tuple[int, int]): point de départ de l'individue
        """
        self.mouvements: 'list[int]' = mouvements_effectues
        self.start_point: tuple[int, int] = start_point
        self.score: int | None = None

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
        path = [self.start_point]
        for move in self.mouvements:
            last_pos = path[-1]
            add = Individue.TEMPLATE_MOVE[move]
            next_x, next_y = last_pos[0]+add[0], last_pos[1]+add[1]
            if 0 > next_x or next_x >= maze.size or 0 > next_y or next_y >= maze.size: 
                next_x, next_y = last_pos
            
            elif maze.get_pixels(next_x, next_y) == MazeColor.WALL:
                next_x, next_y = last_pos
                
            path.append((next_x, next_y))

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
                if maze.get_pixels(vx, vy)[0] != MazeColor.WALL:
                    nb_chemin_possible += 1
            
            # on place un phéromone
            if nb_chemin_possible == 1 and maze.get_pixels(x, y)[0] != MazeColor.WALL and (x, y) != maze.start_coords and (x, y) != maze.end_coords:
                maze.set_wall(x, y)
                idx -= 1
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
        
        assert len(new_movement) <= Population.LONGUEURS_MAX, "la longueur de l'individue est trop grande après fusion"
        return Individue(new_movement, individue1.start_point)
    


    def fitness(self, maze: 'Maze') -> int:
        """Calcule le score de fitness de l'individu"""
        score = 0
        x, y = self.start_point
        self.place_pheromone(maze)
        penalities = {
            "sortie de terrain": 10,
            "foncer dans un mur": 10,
            "bonus d'arrivee": 0,
            "position final": 1,
        }
        

        for step in self.mouvements:
            maze.add_exploration(x, y)
            next_pos = (x+Individue.TEMPLATE_MOVE[step][0], y+Individue.TEMPLATE_MOVE[step][1])
            # print(f"move : {step} => {Individue.TEMPLATE_MOVE[step]}")

            if not(0 <= next_pos[0] < maze.size and 0 <= next_pos[1] < maze.size):
                # le robot sort du labyrinthe
                score += penalities["sortie de terrain"]
                next_pos = (x, y)
            elif maze.get_pixels(x, y)[0] == MazeColor.WALL:
                score += penalities["foncer dans un mur"]
                next_pos = (x, y)
            elif (x, y) == maze.end_coords:
                score += penalities["bonus d'arrivee"]
            
            x, y = next_pos
        
        self.score = score + penalities["position final"]
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





