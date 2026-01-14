import os
from MazeMaker import Maze
from Genetique import Population, Individue
from tests_genetique import courbe_loss

import matplotlib.pyplot as plt


import numpy as np

def moyenne_listes(listes):
    """
    Calcule la moyenne élément par élément de plusieurs listes.
    
    Args:
        listes: Liste de listes d'entiers (toutes de même taille)
    
    Returns:
        Liste contenant la moyenne de chaque élément
    """
    return np.mean(listes, axis=0).tolist()

def plot_curves(data_dict, save_path):
    """
    Sauvegarde des courbes à partir d'un dictionnaire.
    
    Args:
        data_dict: Dictionnaire avec clés str et valeurs liste d'int
        save_path: Chemin complet avec nom de fichier (ex: "output/courbes.png")
    """
    plt.figure(figsize=(10, 6))
    
    for label, values in data_dict.items():
        plt.plot(values, label=label)
    
    plt.xlabel('Index')
    plt.ylabel('Valeur')
    plt.title(save_path.split('/')[-1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure sauvegardée : {save_path}")

def test_hyperparameter(param_name, param_values, setter_func, reset_value, maze, repeats=1):
    """
    Teste différentes valeurs d'un hyperparamètre et génère les courbes.
    
    Args:
        param_name: Nom du paramètre (ex: "selection", "generation")
        param_values: Liste des valeurs à tester
        setter_func: Fonction pour définir le paramètre (ex: Population.set_gardee)
        reset_value: Valeur par défaut à rétablir après les tests
        maze: L'objet maze à utiliser pour la simulation
    """
    scores_min = {}
    
    for value in param_values:
        sum_scores_min = []
        for i in range(repeats):
            setter_func(value)
            population = Population(maze)
            score_min, score_max, score_avg, dist = population.simulation(picture_each_x_generation=population.NOMBRE_GENERATION+1)
            sum_scores_min.append(score_min)
            # Sauvegarde des courbes individuelles
            # courbe_loss(score_min, f"tests/hyperparametres/{param_name}_{value}_repeat{i}_loss_min.png", labels=f"loss min {param_name} {value}")
            # courbe_loss(score_avg, f"tests/hyperparametres/{param_name}_{value}_repeat{i}_loss_avg.png", labels=f"loss avg {param_name} {value}")
            # courbe_loss(score_max, f"tests/hyperparametres/{param_name}_{value}_repeat{i}_loss_max.png", labels=f"loss max {param_name} {value}")
            
        scores_min[str(value)] = moyenne_listes(sum_scores_min)
    
    # Sauvegarde de la comparaison
    plot_curves(scores_min, save_path=f"tests/hyperparametres/{param_name}_comparaison_loss_min.png")
    
    # Réinitialise la valeur par défaut
    setter_func(reset_value)


if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)
    os.makedirs("tests/hyperparametres", exist_ok=True)
    
    path = "tests/hyperparametres"
    files = os.listdir(path)
    for file in files:
        os.remove(f"{path}/{file}")


    maze = Maze(100, 20)

    # Paramètres par défaut
    taux_selection = 0.3
    nombre_generations = 600
    nombre_individues = 900
    longueur_max = 1000
    taux_mutation = 0.4
    offset = 0.20

    # Nombre de répétitions par configuration
    repeats = 3

    Population.set_gardee(taux_selection)
    Population.set_generation(nombre_generations)
    Population.set_individues(nombre_individues)
    Population.set_longueurs_max(longueur_max)
    Population.set_mutation(taux_mutation)
    Individue.set_offset(offset)

    # Tests taux_selection
    test_hyperparameter("selection", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 
                        Population.set_gardee, taux_selection, maze, repeats)
    
    # Tests nombre generations
    test_hyperparameter("generation", [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                        Population.set_generation, nombre_generations, maze, repeats)
    
    # Tests nombre individus
    test_hyperparameter("individues", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                        Population.set_individues, nombre_individues, maze, repeats)
    
    # Tests longueur max
    test_hyperparameter("longueur", [600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 
                        Population.set_longueurs_max, longueur_max, maze, repeats)
    
    # Tests taux mutation
    test_hyperparameter("mutation", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 
                        Population.set_mutation, taux_mutation, maze, repeats)
    
    # Tests offset
    test_hyperparameter("offset", [0.05, 0.1, 0.15, 0.2], 
                        Individue.set_offset, offset, maze, repeats)