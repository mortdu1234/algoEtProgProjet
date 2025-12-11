
from sysconfig import get_path
from Maze_maker import Maze
from random import randint as rdm   
import random

from logs import Logs

LOGS_FILE = Logs()

class Population:
    """Gestion de la population
    """
    LONGUEURS: int = 5 # longueurs des chemins des individues
    TAUX_MUTATION: float = 0.2 # taux de mutation dans les chemins des individues
    TAUX_GARDEE: float = 0.3 # taux des meilleurs individues gardé 
    NOMBRE_INDIVIDUES: int = 100 # nombre total d'individues
    NOMBRE_GENERATION: int = 10 # nombre de génération d'individues
    
    @classmethod
    @staticmethod
    def set_longueurs(new_longueurs: int):
        assert isinstance(new_longueurs, int) and new_longueurs > 0
        Population.LONGUEURS = new_longueurs
    @classmethod
    @staticmethod
    def set_mutation(new_mutation: float):
        assert isinstance(new_mutation, float) and 0 <= new_mutation <= 1
        Population.TAUX_MUTATION = new_mutation
    @classmethod
    @staticmethod
    def set_gardee(new_gardee: float):
        assert isinstance(new_gardee, float) and 0 <= new_gardee <= 1
        Population.TAUX_GARDEE = new_gardee
    @classmethod
    @staticmethod
    def set_individues(nb_individues: int):
        assert isinstance(nb_individues, int) and nb_individues > 1
        Population.NOMBRE_INDIVIDUES = nb_individues
        
    @classmethod
    @staticmethod
    def set_generation(nb_generation: int):
        assert isinstance(nb_generation, int) and nb_generation > 0
        Population.NOMBRE_GENERATION = nb_generation
    

    def log_individues(self):
        """affiche les individues dans les logs"""
        LOGS_FILE.add(" "*(Population.LONGUEURS*3//2-2)+"path"+" "*(Population.LONGUEURS*3//2-2)+" | score\n")
        for individue in self.individues:
            LOGS_FILE.add(
                f"{individue.mouvements} | {individue.score}\n"
            )
    
    def log_statistiques(self):

        min = self.individues[0].score
        max = self.individues[-1].score
        moy = sum(ind.score for ind in self.individues) / len(self.individues)
        
        LOGS_FILE.add(
            "="*100 + "\n"
            +"Statistiques\n"
            +"="*100 + "\n"
            +"parametres:\n"
            +f"\tScore Minimale : {min}\n"
            +f"\tScore Maximale : {max}\n"
            +f"\tScore Moyen    : {moy}\n"
            +"="*100 + "\n"
        )
          
    def log_presentation(self, indice):
        LOGS_FILE.add(
            "="*100 + "\n"
            +f"{indice}\n"
            +"="*100 + "\n"
            +"parametres:\n"
            +f"\tLongueur des chemins : {Population.LONGUEURS}\n"
            +f"\tNombre de générations : {Population.NOMBRE_GENERATION}\n"
            +f"\tNombre d'individues par génération : {Population.NOMBRE_INDIVIDUES}\n"
            +f"\tTaux d'individue gardé entre deux générations : {Population.TAUX_GARDEE}\n"
            +f"\tTaux de mutation des individues : {Population.TAUX_MUTATION}\n"
            +"="*100 + "\n"
        )
    
    def log_explored_map(self, maze: 'Maze', indice: int):
        map = maze.pixels.copy()
        for individu in self.individues:
            for x, y in individu.get_path(maze):
                map[x][y] = 4
        fig = maze.convert_to_image(map, "Exploration")
        fig.savefig(f"images/exploration/exploration_map{indice}.png")





    def __init__(self, maze: 'Maze'):
        # Population initiale
        LOGS_FILE.new()
        self.individues: list[Individue] = []
        self.maze: 'Maze' = maze

        for _ in range(Population.NOMBRE_INDIVIDUES):
            individue: Individue = Individue([rdm(0, 7) for _ in range(Population.LONGUEURS)], maze.start_coords)
            self.individues.append(individue)


    def __str__(self):
        res = ""
        for idv in self.individues:
            res += f"{idv}\n"
        return res


    def simulation(self):
        self.calcul_fitness()
        self.tri_individue()

        self.log_presentation("Initialisation")
        self.log_individues()

        
        

        
        for i in range(Population.NOMBRE_GENERATION):
            self.selection()
            self.reproduction()
            self.mutation()
            self.calcul_fitness()
            self.tri_individue()
            
            self.log_presentation(f"Generation {i}")
            self.log_statistiques()
            self.log_explored_map(self.maze, i)

        self.log_individues()
            

    def calcul_fitness(self):
        for individu in self.individues:
            if individu.score is None:
                individu.fitness(self.maze)

    def tri_individue(self):
        """effectue un tri des individues de celui qui as le score le plus faible a celui qui a le score le plus grand
        """
        self.individues.sort(key=lambda ind: ind.score)

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
        nb_mutant = round(Population.NOMBRE_GENERATION * Population.TAUX_MUTATION)
        nb_mutation = round(Population.LONGUEURS * Population.TAUX_MUTATION)
        idx_mutants = []

        for _ in range(nb_mutant):
            # selection du mutant
            idx_mutant = random.randint(0, Population.NOMBRE_GENERATION-1)
            while idx_mutant in idx_mutants:
                idx_mutant = random.randint(0, Population.NOMBRE_GENERATION-1)
            idx_mutants.append(idx_mutant)
            mutant : Individue = self.individues[idx_mutant]
            # print(f"modification de l'individue {mutant}")

            # selection des mutations du mutants
            idx_mutations = []
            for _ in range(nb_mutation):
                idx_mutation = random.randint(0, Population.LONGUEURS-1)
                while idx_mutation in idx_mutations:
                    idx_mutation = random.randint(0, Population.LONGUEURS-1)
                idx_mutations.append(idx_mutation)

                new_value = random.randint(0, 7)
                # print(f"\tidx:{idx_mutation} | {mutant.mouvements[idx_mutation]} -> {new_value}")
                mutant.mouvements[idx_mutation] = new_value
            # print(f"individue modifié           {mutant}")
            





class Individue:
    TEMPLATE_MOVE = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)] # liste des mouvements possibles
    CUT_OFFSET = 0.05 # % d'offset du cut autour du milieux
    
    def __init__(self, mouvements_effectues: 'list[int]', start_point: 'tuple[int, int]'):
        """gestion des individues

        Args:
            mouvements_effectues (list[int]): liste des mouvements effectuer par l'individue
            start_point (tuple[int, int]): point de départ de l'individue
        """
        self.mouvements: 'list[int]' = mouvements_effectues
        self.start_point: tuple[int, int] = start_point
        self.score: int = None

    def __str__(self):
        return f"{self.mouvements} | score : {self.score}"
    
    def get_path(self, maze: 'Maze') -> 'list[tuple[int, int]]':
        """récupère la liste des case visité par l'individue

        Returns:
            list[tuple[int, int]]: liste des coordonnées visités
        """
        path = [self.start_point]
        for move in self.mouvements:
            # print(f"effectue le mouvement {move}")
            last_pos = path[-1]
            add = Individue.TEMPLATE_MOVE[move]
            next_case = (last_pos[0]+add[0], last_pos[1]+add[1])
            # print(f"{last_pos} => {next_case}")
            if 0 < next_case[0] >= maze.size or 0 < next_case[1] >= maze.size: 
                next_case = last_pos
                # print("cas1")
            elif maze.pixels[next_case] == 0: # si la case suivante est un mur alors on ignore le mouvement
                next_case = last_pos
                # print("cas2")

            path.append(next_case)
        return path
    
    @staticmethod
    def fusion(individue1: 'Individue', individue2: 'Individue'):
        longueur = len(individue1.mouvements)
        offset = round(longueur*Individue.CUT_OFFSET)
        milieu = longueur//2
        pos_cut = random.randint(milieu-offset, milieu+offset) 
        
        new_movement = individue1.mouvements[:pos_cut] + individue2.mouvements[pos_cut:]
        return Individue(new_movement, individue1.start_point)
    
    def fitness(self, maze: 'Maze') -> int:
        """détermine a quel point l'individue est adapté
        plus le score est bas, mieux c'est

        Args:
            maze (Maze): le labyrinthe dans lequel il se balade

        Returns:
            int: score de l'individue
        """
        score = 0
        # donne le score en fonction de la distance au point (distance physique)
        
        # pour chaque mouvement, vérifie si c'est le choix optimal, alors récompense, sinon pénalise
        x, y = self.start_point
        for step in self.mouvements:
            if step == maze.dimention_map[x, y]:
                score -= 1
            else:
                score += 1
            
            x, y = x+Individue.TEMPLATE_MOVE[step][0], y+Individue.TEMPLATE_MOVE[step][1]
            # si le robot vas dans un mur, annule le mouvement et pénalise fortement
            if not(0 <= x < maze.size and 0 <= y < maze.size):
                score += 10
                x, y = x-Individue.TEMPLATE_MOVE[step][0], y-Individue.TEMPLATE_MOVE[step][1]

            elif maze.pixels[x][y] == 0:
                score += 10
                x, y = x-Individue.TEMPLATE_MOVE[step][0], y-Individue.TEMPLATE_MOVE[step][1]

        # pénalité de fin de position
        score += maze.dijkstra_map[x][y]

        self.score = score
        return score


    
            
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

if __name__ == "__main__":
    maze = Maze(20, 20)
    maze.set_up_maze()
    maze.apply_dijkstra()
    maze.set_up_dimentionnal_map()

    # DEBUG
    # best_path = maze.set_path()
    # best_path_robot = get_chemin_robot(best_path)
    ########

    Population.set_longueurs(300)
    Population.set_generation(10)
    Population.set_individues(10)
    Population.set_gardee(0.3)
    Population.set_mutation(0.5)


    pop = Population(maze)
    pop.simulation()





    # pop.calcul_fitness()
    # pop.tri_individue()
    # print(pop)
    # pop.selection()
    # print(pop)
    # pop.reproduction()
    # print(pop)
    # pop.mutation()
    # print(pop)




