from itertools import count



class Logs:
    def __init__(self, filename: str="logs"):
        self.file = f"logs/{filename}.txt"
        with open(f"{self.file}", "w") as file:
            pass 
    
    def write(self, line: str):
        with open(f"{self.file}", "a") as file:
            file.write(line + "\n")

    def write_generation(self, maze, best_individue, first_generation: int, last_generation: int):
        best_path = best_individue.get_path(maze)
        key = "bonus d'arrivee"
        
        res = "=" * 50 + "\n"
        res += f"GENERATION {last_generation} - {first_generation}\n"
        res += "=" * 50 + "\n"
        res += f"\t taille du chemin le plus court réalisé/theorique: {len(best_path)}/{maze.get_pixels(maze.start_coords[0], maze.start_coords[1])[1]}\n"
        res += f"\t nombre de mur prit : {best_individue.counter['foncer dans un mur']} * {best_individue.penalities['foncer dans un mur']}\n"
        res += f"\t nombre de sortie de terrain : {best_individue.counter['sortie de terrain']} * {best_individue.penalities['sortie de terrain']}\n"
        res += f"\t bonus d'arrivée : {best_individue.counter[key]} * {best_individue.penalities[key]}\n"
        res += f"\t position finale : {best_individue.counter['position final']} ** {best_individue.penalities['position final']}\n"
        res += f"\t nombre de retours en arriere : {best_individue.counter['retour en arriere']} * {best_individue.penalities['retour en arriere']}\n"
        res += "=" * 50 + "\n"
        res += f"score : {best_individue.score}\n"
        res += f"score : {best_individue.scoreExplication}\n"
        res += "=" * 50 + "\n"
        res += "CHEMIN EMPRUNTE\n"
        res += "=" * 50 + "\n"
        res += f"{self.zip_path(best_path)}\n"
        res += "=" * 50 + "\n"
        self.write(res)
        


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

        