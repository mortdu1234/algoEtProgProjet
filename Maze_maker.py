from math import pi
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Maze:
    """Gestion du labyrinthe
    0 : mur
    1 : chemin
    2 : Goal
    3 : Départ
    """
    def __init__(self, size: int, seed: int=0):
        """Initialisation du labyrinthe

        Args:
            size (int): taille du labyrinthe de taille (sizeXsize)
            seed (int, optional): seed d'aléatoire pour la création du labyrinthe. 
                                Une valeur de 0 prend une seed aléatoire généré automatiquement. Defaults to 0.
        """
        self.seed: int = seed
        self.random = random(seed)
        if self.seed != 0:
            random.seed(self.seed)
        self.size: int = size
        self.pixels = np.zeros(size**2).reshape(size, size)
        self.dijkstra_map = None
        self.dimention_map = np.zeros(size**2).reshape(size, size)
        self.path_map = None
        self.start_coords: tuple[int, int] = None
        self.end_coords: tuple[int, int] = None
    
    def get_maze_from_image(self, image_path: str):
        """création d'un labyrinthe à partir d'une image

        Args:
            image_path (str): path vers l'image
        """
        img = Image.open(image_path)
        img_pixels = np.array(img)
        print(img_pixels)
        for x, line in enumerate(img_pixels):
            for y, pixel in enumerate(line):
                match tuple(pixel):
                    case (0, 0, 0):  # noir | mur
                        self.pixels[x][y] = 0
                    case (255, 255, 255): # blanc | chemin
                        self.pixels[x][y] = 1
                    case (255, 0, 0):  # rouge | arrivée
                        self.pixels[x][y] = 2
                    case (0, 255, 0): # vert | départ
                        self.pixels[x][y] = 3

    def __str__(self):
        """Représentation du labyrinthe en version terminal

        Returns:
            str:
        """
        res = f"Labyrinthe de taille {self.size}x{self.size}\n"
        res += str(self.pixels)
        return res + "\n"
    
    def _convert_to_rgb(self, pixels):
        """Convertie une matrice de pixels avec des valeurs entières
        en une matrice de pixels RGB

        Args:
            pixels (np.array): matrice de pixels avec des valeurs entières

        Returns:
            np.array: matrice des pixels avec des valeurs RGB
        """
        colormap = np.array([
            [0,   0,   0],      # 0 : mur
            [255, 255, 255],    # 1 : chemin
            [255, 0, 0],        # 2 : goal
            [0, 255, 0],        # 3 : départ
            [255, 255, 0]       # 4 : exploration
        ], dtype=np.uint8)

        rgb_pixels = colormap[pixels.astype(int)]
        return rgb_pixels
            
    def _est_eligible(self, x: int, y: int) -> bool:
        """Vérifie si une case est éligible ou non

        Args:
            x (int): position x de la case
            y (int): position y de la case

        Returns:
            bool: éligibilité de la case
        """
        sum = 0
        for vx, vy in self._get_voisins(x, y):
            if self.pixels[vx][vy] != 0:
                sum += 1
        
        return self.pixels[x][y] == 0 and sum == 1

    def _get_voisins(self, x: int, y: int) -> list[tuple[int, int]]:
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

    def set_up_maze(self):
        """Initialisation du labyrinthe
        """
        rx = random.randint(0, self.size-1)
        ry = random.randint(0, self.size-1)
        self.pixels[rx][ry] = 1
        pile = [(rx, ry)]
        
        while len(pile) > 0:
            x, y = pile[-1]
            
            # Trouver les voisins éligibles
            voisins_eligibles = []
            for vx, vy in self._get_voisins(x, y):
                if self._est_eligible(vx, vy):
                    voisins_eligibles.append((vx, vy))
            
            if voisins_eligibles:
                nx, ny = random.choice(voisins_eligibles)
                self.pixels[nx][ny] = 1
                pile.append((nx, ny))
            else:
                pile.pop()

        # set du goal
        # augmente uniquement le x pour le choix de l'arrivée
        x, y = 0, 0
        pixel_values = self.pixels[x][y]
        while pixel_values == 0:
            x += 1
            pixel_values = self.pixels[x][y]    
        self.end_coords = (x, y)  
        self.pixels[x][y] = 2

        # set du départ
        # réduit uniquement le x pour le choix du départ
        x, y = self.size-1, self.size-1
        pixel_values = self.pixels[x][y]
        while pixel_values == 0:
            x -= 1
            pixel_values = self.pixels[x][y]  
        self.start_coords = (x, y)
        self.pixels[x][y] = 3
        
    def apply_dijkstra(self, x: int = None, y: int = None):
        """applique l'algorithme de Dijkstra sur le labyrinthe et l'applique dans l'attribut
        self.dijkstra_map

        Args:
            x (int): coordonnée x du point d'arrivée
            y (int): coordonnée y du point d'arrivée
        """
        if x or y is None:
            x, y = self.end_coords 
        
        color_map = np.array([
            -np.inf,   # 0 : mur  
            np.inf,    # 1 : chemin (non visité)
            np.inf,    # 2 : goal (non visité)
            np.inf,    # 3 : départ (non visité)
        ])


        self.dijkstra_map = color_map[self.pixels.astype(int)]
        self.dijkstra_map[x][y] = 0 # point d'arrivée
        to_check = [(x, y)]

        while len(to_check) > 0:
            x, y = to_check.pop(0)
            for vx, vy in self._get_voisins(x, y):
                cell = self.dijkstra_map[vx][vy]
                if cell == np.inf:
                    self.dijkstra_map[vx][vy] = self.dijkstra_map[x][y] + 1
                    to_check.append((vx, vy))

        
    def convert_to_image(self, pixels, title="Maze"):
        """Création d'une figure en fonction

        Args:
            pixels (_type_): matrice de pixel utilisé pour l'image
            title (str, optional): nom attitré à la figure. Defaults to "Maze".

        Returns:
            Figure: la figure 
        """
        rgb_pixels = self._convert_to_rgb(pixels)
        
        fig = plt.figure(figsize=(8, 8)) 
        plt.imshow(rgb_pixels)
        plt.title(title)
        plt.draw()
        plt.pause(0.001) 
        return fig 

    def show_dijkstra_map(self):
        """Affiche la map de dijkstra
        si la size du labyrinthe alors la valeurs de dijkstra sera affiché sur chaque pixel

        Returns:
            Figure: 
        """
        fig = plt.figure(figsize=(10, 10))  
        plt.imshow(self.dijkstra_map, cmap='plasma', interpolation='nearest')
        
        # ajout des valeurs dans chaque case UNIQUEMENT pour un petit labyrinthe
        if self.size < 20:
            for i in range(self.size):
                for j in range(self.size):
                    value = self.dijkstra_map[i][j]
                    text = '' if value == -np.inf else f'{int(value)}'
                    color = 'white' if value > 20 or value == -np.inf else 'white'
                    
                    plt.text(j, i, text, ha='center', va='center', 
                            color=color, fontsize=8, fontweight='bold')
            
        plt.colorbar(label='Distance')
        plt.title('Carte Dijkstra')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        return fig

    def set_up_dimentionnal_map(self):
        """crée une map dimentionnelle avec le meilleur mouvement dans chaque cases
        """
        assert self.dijkstra_map is not None
        TEMPLATE_MOVE = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

        for x in range(self.size):
            for y in range(self.size):
                case = self.dijkstra_map[x][y]
                if case <= 0:
                    self.dimention_map[x][y] = None
                else:
                    for idx, coords in enumerate(TEMPLATE_MOVE):
                        vx, vy = coords
                        if 0 <= x+vx < self.size and 0 <= y+vy < self.size: 
                            next_case = self.dijkstra_map[x+vx][y+vy]
                            # print(case, next_case)
                            if next_case == round(case - 1, 1):
                                self.dimention_map[x][y] = idx

            

    def set_path(self) -> list[tuple[int, int]]:
        """trace le chemin du start au end en rouge (goal)

        Returns:
            list[tuple[int, int]]: path utilisé avec la liste de chaque coordonnée parcourus
        """
        end_x, end_y = self.end_coords
        start_x, start_y = self.start_coords
        
        self.path_map = np.copy(self.pixels)
        if self.pixels[end_x][end_y] != 2:
            print("End point is not a goal.")
            return
        
        if self.pixels[start_x][start_y] == 0:
            print("Start point is a wall.")
            return
        
        if self.dijkstra_map is None:
            print("Dijkstra map not computed yet.")
            return
        
        x, y = start_x, start_y
        self.path_map[x][y] = 2  # marquer le chemin
        path = [(x, y)]
        while (x, y) != (end_x, end_y):
            current_value = self.dijkstra_map[x][y]
            for vx, vy in self._get_voisins(x, y):
                if 0 <= self.dijkstra_map[vx][vy] < current_value:
                    x, y = (vx, vy)
                    path.append((x, y))
                    self.path_map[x][y] = 2  # marquer le chemin
                    break
        return path
            

if __name__ == "__main__":
    """    
    plt.ion()  # Active le mode interactif
    
    maze = Maze(10, 30)
    maze.set_up_maze()
    
    fig1 = maze.convert_to_image(maze.pixels, "Labyrinthe initial")
    fig1.savefig("images/.png")
    
    maze.apply_dijkstra()
    fig2 = maze.show_dijkstra_map()
    
    path = maze.set_path()
    fig3 = maze.convert_to_image(maze.path_map, "Chemin trouvé")
    print(f"le chemin fait : {len(path)}\n{path}")
    plt.show(block=True) 
    """

    # pour enregistrer les images
    a = Maze(10)
    a.set_up_maze()
    a.apply_dijkstra()
    a.set_up_dimentionnal_map()
    
    for line in a.dijkstra_map:
        print(line)
    fig = a.show_dijkstra_map()
    

    for line in a.dimention_map:
        print(line)

    plt.show(block=True) 

    """
    for i in [5, 50, 500]:
        maze = Maze(i, 30)
        maze.set_up_maze()     
        fig1 = maze.convert_to_image(maze.pixels, f"Labyrinthe initial {i}x{i}")
        fig1.savefig(f"images/labyrinthe_initial_{i}x{i}.png")
    """