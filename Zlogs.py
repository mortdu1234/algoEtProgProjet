from itertools import count

from Zgenetique2 import Individue


class Logs:
    def __init__(self, filename: str="logs"):
        self.file = f"logs/{filename}.txt"
        with open(f"{self.file}", "w") as file:
            pass 
    
    def write(self, line: str):
        with open(f"{self.file}", "a") as file:
            file.write(line + "\n")

    def write_generation(self, maze: 'Maze', best_individue: 'Individue', first_generation: int, last_generation: int):
        best_path = best_individue.get_path(maze)
        res = "=" * 50 + "\n"
        res += f"GENERATION {last_generation} - {first_generation}\n"
        res += "=" * 50 + "\n"
        res += f"\t taille du chemin le plus court : {len(best_path)}\n"
        res += f"\t"

    def zip_path(self, path:list[tuple[int, int]])-> list[tuple[int, int, int]]:
        current_x, current_y = path[0]
        counter = 0
        res = []
        for x, y in path:
            if x==current_x and y==current_y:
                counter += 1
            else:
                res.append((current_x, current_y, counter))
                counter = 1
                current_x = x
                current_y = y

        res.append((current_x, current_y, counter))
        return res  

    
if __name__ == "__main__":
    l = Logs("t")

    liste1 = [(1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (1, 2)]
    liste2 = l.zip_path(liste1)
    print(liste2)

        