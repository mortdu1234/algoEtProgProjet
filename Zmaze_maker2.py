from enum import Enum
import random
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class MazeColor(Enum):
    WALL = (0, 0, 0)
    PATH = (255, 255, 255)
    GOAL = (255, 0, 0)
    START = (0, 255, 0)
    EXPLORATION = (255, 255, 0)
    BEST_PATH = (0, 255, 255)

class Maze:
    def __init__(self, size: int, seed: int = 0):
        self.size = size
        self.random = random.Random(seed)
        self.pixels_map: list[list[MazeColor]] = [[MazeColor.WALL for _ in range(self.size)] for _ in range(self.size)]
        self.dijkstra_map = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.dimention_map = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.explored_map = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.explored_phase_map = [[0 for _ in range(self.size)] for _ in range(self.size)]

        self.start_coords = (size - 1, size - 1)
        self.end_coords = (0, 0)
        
        self.__generate_maze()
        self.__generate_dijkstra()
        self.__generate_dimentionnal_map()

    
    def get_voisins(self, x: int, y: int) -> list[tuple[int, int]]:
        """Renvois la liste des voisins d'une case
        les sorties de liste (bords) ne sont juste pas prit en comptes
        le pixel d'origine n'est pas dans les voisins

        Args:
            x (int): postiion x de la case
            y (int): position y de la case

        Returns:
            list[tuple[int, int]]: liste des coordonnées de cases voisines
        """
        res = []
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == 0 and j == 0: 
                    continue
                dx = x + i
                dy = y + j
                if (0 <= dx < self.size) and (0 <= dy < self.size):
                    res.append((dx, dy))
        return res
    
    def __est_eligible(self, x: int, y: int) -> bool:
        """Vérifie si une case est éligible ou non

        Args:
            x (int): position x de la case
            y (int): position y de la case

        Returns:
            bool: éligibilité de la case
        """
        sum = 0
        for vx, vy in self.get_voisins(x, y):
            if self.pixels_map[vx][vy] != MazeColor.WALL:
                sum += 1
        
        return self.pixels_map[x][y] == MazeColor.WALL and sum == 1
    
    def __generate_maze(self):
        """Initialisation du labyrinthe
        """


        rx = self.random.randint(0, self.size-1)
        ry = self.random.randint(0, self.size-1)
        self.pixels_map[rx][ry] = MazeColor.PATH
        pile = [(rx, ry)]
        
        while len(pile) > 0:
            x, y = pile[-1]
            
            # Trouver les voisins éligibles
            voisins_eligibles = []
            for vx, vy in self.get_voisins(x, y):
                if self.__est_eligible(vx, vy):
                    voisins_eligibles.append((vx, vy))
            if voisins_eligibles:
                nx, ny = self.random.choice(voisins_eligibles)
                self.pixels_map[nx][ny] = MazeColor.PATH
                pile.append((nx, ny))
            else:
                pile.pop()

        # set du goal
        # augmente uniquement le x pour le choix de l'arrivée
        x, y = 0, 0
        pixel_values = self.pixels_map[x][y]
        while pixel_values == MazeColor.WALL:
            x += 1
            pixel_values = self.pixels_map[x][y]    
        self.end_coords = (x, y)  
        self.pixels_map[x][y] = MazeColor.GOAL

        # set du départ
        # réduit uniquement le x pour le choix du départ
        x, y = self.size-1, self.size-1
        pixel_values = self.pixels_map[x][y]
        while pixel_values == MazeColor.WALL:
            x -= 1
            pixel_values = self.pixels_map[x][y]  
        self.start_coords = (x, y)
        self.pixels_map[x][y] = MazeColor.START

    def __generate_dijkstra(self):
        """_summary_
        """
        value_max = self.size**2

        for i in range(self.size):
            for j in range(self.size):
                if self.pixels_map[i][j] == MazeColor.WALL:
                    self.dijkstra_map[i][j] = -1
                else:
                    self.dijkstra_map[i][j] = value_max

        self.dijkstra_map[self.end_coords[0]][self.end_coords[1]] = 0 # point d'arrivée
        to_check = [self.end_coords]

        while len(to_check) > 0:
            x, y = to_check.pop(0)
            for vx, vy in self.get_voisins(x, y):
                cell = self.dijkstra_map[vx][vy]
                if cell == value_max:
                    self.dijkstra_map[vx][vy] = self.dijkstra_map[x][y] + 1
                    to_check.append((vx, vy))

    def __generate_dimentionnal_map(self):
        """crée une map dimentionnelle avec le meilleur mouvement dans chaque cases
        """
        assert self.dijkstra_map is not None
        TEMPLATE_MOVE = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

        for x in range(self.size):
            for y in range(self.size):
                case = self.dijkstra_map[x][y]
                if case <= 0:
                    self.dimention_map[x][y] = -1
                else:
                    for idx, coords in enumerate(TEMPLATE_MOVE):
                        vx, vy = coords
                        if 0 <= x+vx < self.size and 0 <= y+vy < self.size: 
                            next_case = self.dijkstra_map[x+vx][y+vy]
                            # print(case, next_case)
                            if next_case == round(case - 1, 1):
                                self.dimention_map[x][y] = idx
  
    def get_pixels(self, x: int, y: int) -> tuple[MazeColor, int, int]:
        """retourne la valeur du pixel x, y pour toutes les maps
        Args:
            x (int): 
            y (int): 
        Returns: tuple[int, int]: map, dijkstra, dimention
        """
        assert(0 <= x < self.size and 0 <= y < self.size), f"Coordonnées hors limites: 0 <= {x} < {self.size}, 0 <= {y} < {self.size}"
        return self.pixels_map[x][y], self.dijkstra_map[x][y], self.dimention_map[x][y]
    
    def reset_exploration(self):
        for i in range(self.size):
            for j in range(self.size):
                self.explored_phase_map[i][j] = 0
        
    def add_exploration(self, x:int, y:int):
        assert(0 <= x < self.size and 0 <= y < self.size), f"Coordonnées hors limites: 0 <= {x} < {self.size}, 0 <= {y} < {self.size}"
        self.explored_map[x][y] += 1
        self.explored_phase_map[x][y] += 1
    
    def get_exploration_phase(self, x:int, y:int):
        """renvois l'exploration a une phase

        Args:
            x (int): _description_
            y (int): _description_

        Returns:
            int: nombre de robot passer sur cette case
        """
        assert(0 <= x < self.size and 0 <= y < self.size), f"Coordonnées hors limites: 0 <= {x} < {self.size}, 0 <= {y} < {self.size}"
        return self.explored_phase_map[x][y]


    def set_wall(self, x:int, y:int):
        """pose un wall au coordonnee x, y"""
        assert(0 <= x < self.size and 0 <= y < self.size), f"Coordonnées hors limites: 0 <= {x} < {self.size}, 0 <= {y} < {self.size}"
        self.pixels_map[x][y] = MazeColor.WALL

    def set_path(self, path: list[tuple[int, int]], color: MazeColor) -> Figure:
        """Crée une figure isolée avec le chemin coloré"""
        # Créer une copie profonde de la map pour ne pas modifier l'original
        pixel_map = [row[:] for row in self.pixels_map]
        
        for x, y in path:
            pixel_map[x][y] = color
        
        # Convertir en tableau RGB pour l'affichage
        import numpy as np
        rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                rgb_array[i][j] = pixel_map[i][j].value
        
        # Créer une figure complètement isolée
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(rgb_array)
        ax.set_title("Path Map")
        ax.axis('off')
        fig.tight_layout()
        
        return fig



    def get_fig_explored_phase_map(self, title: str="", show_values: bool=True, show_image: bool=True) -> Figure:
        tab_np = np.array(self.explored_phase_map)
        tab_mask = np.ma.masked_where(tab_np == -1, tab_np)

        fig, ax = plt.subplots(figsize=(8, 8)) 
        cmap = plt.cm.hot
        cmap.set_bad(color="black")


        if show_image:
            im = ax.imshow(tab_mask, cmap=cmap, interpolation='nearest')
        

        if show_values:
            fig.colorbar(im, ax=ax, label='Exploration')
            for i in range(self.size):
                for j in range(self.size):
                    text = ax.text(j, i, tab_np[i, j],
                        ha="center", va="center",
                        color="white" if tab_np[i, j] == -1 else "black",
                        fontsize=12, weight='bold')
        
        ax.set_title(title)

        return fig




    def get_fig_exploration_map(self, name: str = "Explored Map") -> tuple[Figure, Figure]:
        """Retourne les figures matplotlib de la map explorée (phase et full)"""
        
        ###########################################
        # EXPLORED PHASE MAP
        ###########################################
        fig_explored_phase = plt.figure(figsize=(8, 8))
        ax_phase = fig_explored_phase.add_subplot(111)
        
        # Créer une copie pour éviter les références
        phase_data = np.array(self.explored_phase_map, dtype=float)
        im_phase = ax_phase.imshow(phase_data, cmap='viridis', interpolation='nearest')
        ax_phase.set_title(f"{name} - Phase")
        fig_explored_phase.colorbar(im_phase, ax=ax_phase)
        fig_explored_phase.tight_layout()
        
        ###########################################
        # EXPLORED FULL MAP
        ###########################################
        fig_explored_full = plt.figure(figsize=(8, 8))
        ax_full = fig_explored_full.add_subplot(111)
        
        # Créer une copie pour éviter les références
        full_data = np.array(self.explored_map, dtype=float)
        im_full = ax_full.imshow(full_data, cmap='hot', interpolation='nearest')
        ax_full.set_title(f"{name} - Full")
        fig_explored_full.colorbar(im_full, ax=ax_full)
        fig_explored_full.tight_layout()
        
        return (fig_explored_phase, fig_explored_full)

    def get_fig_of_maze(self) -> tuple[Figure, Figure, Figure]:
        """Renvois les figures matplotlib du labyrinthe (isolées)"""
        
        ###########################################
        # PIXELS MAP
        ###########################################
        fig_pixels_map = plt.figure(figsize=(8, 8))
        ax_pixels = fig_pixels_map.add_subplot(111)
        
        # Convertir la map en RGB
        rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                rgb_array[i][j] = self.pixels_map[i][j].value
        
        ax_pixels.imshow(rgb_array)
        ax_pixels.set_title("Pixels Map")
        ax_pixels.axis('off')
        fig_pixels_map.tight_layout()

        ###########################################
        # DIJKSTRA MAP
        ###########################################
        fig_dijkstra_map = plt.figure(figsize=(8, 8))
        ax_dijkstra = fig_dijkstra_map.add_subplot(111)
        
        # Créer une copie et masquer les murs (-1)
        dijkstra_data = np.array(self.dijkstra_map, dtype=float)
        dijkstra_data[dijkstra_data == -1] = np.nan
        
        im_dijkstra = ax_dijkstra.imshow(dijkstra_data, cmap='viridis', interpolation='nearest')
        
        # Ajout des valeurs uniquement pour petit labyrinthe
        if self.size <= 20:
            for i in range(self.size):
                for j in range(self.size):
                    value = self.dijkstra_map[i][j]
                    if value != -1:
                        ax_dijkstra.text(j, i, f'{int(value)}',
                                ha='center', va='center',
                                color='white', fontsize=8, fontweight='bold')

        ax_dijkstra.set_title('Carte Dijkstra')
        fig_dijkstra_map.colorbar(im_dijkstra, ax=ax_dijkstra)
        fig_dijkstra_map.tight_layout()

        ###########################################
        # DIMENTION MAP
        ###########################################
        fig_dimention_map = plt.figure(figsize=(8, 8))
        ax_dimention = fig_dimention_map.add_subplot(111)
        
        # Créer une image RGB: noir pour les murs, blanc pour les chemins
        dimention_rgb = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255  # Blanc par défaut
        
        for i in range(self.size):
            for j in range(self.size):
                if self.dimention_map[i][j] == -1:
                    dimention_rgb[i][j] = [0, 0, 0]  # Noir pour les murs
        
        ax_dimention.imshow(dimention_rgb, interpolation='nearest')
        
        # Ajout des valeurs uniquement pour petit labyrinthe
        if self.size <= 20:
            for i in range(self.size):
                for j in range(self.size):
                    value = self.dimention_map[i][j]
                    if value != -1:
                        ax_dimention.text(j, i, f'{int(value)}',
                                ha='center', va='center',
                                color='black', fontsize=8, fontweight='bold')
            
        ax_dimention.set_title("Carte Dimention")
        ax_dimention.axis('off')  # Suppression des graduations
        fig_dimention_map.tight_layout()

        return fig_pixels_map, fig_dijkstra_map, fig_dimention_map