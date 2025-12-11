"""création d'un fichier de logs"""

class Logs:
    def __init__(self, path="logs.txt"):
        self.path = path
    
    def add(self, ligne: str):
        with open(self.path, "a") as f:
            f.write(ligne)
        
    def new(self):
        with open(self.path, "w") as f:
            f.write("")
         