# Maze Solver - Algorithme Génétique

## Description du projet

Ce projet implémente un **algorithme génétique** pour résoudre des labyrinthes générés procéduralement. Le système fait évoluer une population d'individus qui tentent de trouver le chemin optimal du point de départ à l'arrivée dans un labyrinthe.

### Caractéristiques principales

- **Génération de labyrinthes** : Création procédurale de labyrinthes de taille variable avec seed contrôlable
- **Résolution par algorithme génétique** : Population d'individus évoluant sur plusieurs générations
- **Calcul de Dijkstra** : Génération automatique de la carte des distances optimales
- **Visualisations multiples** : 
  - Carte du labyrinthe
  - Carte de Dijkstra (distances)
  - Carte dimensionnelle (directions optimales)
  - Carte d'exploration (zones visitées par les individus)
  - Évolution des meilleurs chemins
- **Système de phéromones** : Les individus peuvent fermer des impasses découvertes
- **Logs détaillés** : Suivi de l'évolution génération par génération

## Architecture du projet

```
.
├── MazeMaker.py                                # Génération et gestion des labyrinthes
├── Genetique.py                                # Algorithme génétique (Population, Individue)
├── Logs.py                                     # Système de logs
├── tests_genetique.py                          # Script principal d'exécution
├── tests_MazeMaker.py                          # Tests de génération de labyrinthes
├── tests_MazeMaker_EvaluationsEmpiriques.py    # Évaluations de performance
├── logs/                                       # Dossier des logs générés
│   └── explorations/                           # Visualisations par génération
└── tests/                                      # Dossier des résultats de tests
    ├── genetique/                              # Résultats des simulations génétiques
    └── hyperparametres/                        # Résultats des tests des hyperparametres

```
Les différents dossier sont générés automatiquement s' ils sont manquant

## Installation

### Prérequis

- Python 3.10.12
- Bibliothèques requises :
```bash
pip install -r requirement.txt
```

## Utilisation

### Exécution basique

Pour lancer une simulation génétique sur un labyrinthe :

```bash
python tests_genetique.py
```

Ce script va :
1. Générer un labyrinthe de 20×20
2. Lancer une simulation génétique avec les paramètres par défaut
3. Créer des visualisations dans `tests/genetique/`
4. Générer des logs d'exploration dans `logs/explorations/`

### Configuration des paramètres

Dans `tests_genetique.py`, vous pouvez modifier les paramètres de l'algorithme génétique :

```python
# Paramètres de la population
Population.set_gardee(0.02)              # Taux de conservation des meilleurs (2%)
Population.set_generation(5000)          # Nombre de générations
Population.set_individues(200)           # Taille de la population
Population.set_longueurs_max(1000)       # Longueur max des chemins
Population.set_mutation(0.9)             # Taux de mutation (90%)

# Paramètres des individus
Individue.set_offset(0.10)               # Offset pour le croisement (10%)
```

### Paramètres du labyrinthe

```python
# Créer un labyrinthe
maze = Maze(20)           # Labyrinthe 20×20 avec seed aléatoire
maze = Maze(50, 42)       # Labyrinthe 50×50 avec seed 42
```

### Génération de visualisations

Le script génère automatiquement plusieurs types de visualisations :

```python
# Carte initiale du labyrinthe
fig = maze.get_fig_pixel_map(title="Labyrinthe", show_image=True)
fig.savefig("labyrinthe.png")

# Carte de Dijkstra (distances optimales)
fig = maze.get_fig_dijkstra_map(title="Dijkstra", show_values=True)
fig.savefig("dijkstra.png")

# Carte d'exploration
fig = maze.get_fig_explored_full_map(title="Exploration", show_values=True)
fig.savefig("exploration.png")

# Meilleur chemin trouvé
best_path = individue.get_path(maze)
fig = maze.set_path(best_path, MazeColor.BEST_PATH, MazeColor.BEST_FINISH)
fig.savefig("best_path.png")
```

### Tests et évaluations

Pour tester la génération de labyrinthes et évaluer les performances :

```bash
python tests_MazeMaker.py
```

Ce script génère :
- Des labyrinthes de différentes tailles (20×20, 100×100, 500×500)
- Les cartes de Dijkstra et dimensionnelles correspondantes
- Les meilleurs chemins théoriques

## Fonctionnement de l'algorithme génétique

### 1. Initialisation
Une population d'individus est créée avec des chemins aléatoires de mouvements (0-7 représentant 8 directions).

### 2. Évaluation (Fitness)
Chaque individu est évalué selon plusieurs critères :
- **Position finale** : Distance à l'arrivée (Dijkstra)
- **Sorties de terrain** : Tentatives de sortir du labyrinthe
- **Collisions avec les murs** : Nombre de fois où l'individu heurte un mur
- **Retours en arrière** : Passages répétés sur les mêmes cases
- **Bonus d'arrivée** : Récompense si l'arrivée est atteinte

### 3. Sélection
Seuls les meilleurs individus (défini par `TAUX_GARDEE`) sont conservés.

### 4. Reproduction
Les individus restants se reproduisent par croisement (crossover) pour recréer une population complète.

### 5. Mutation
Un pourcentage d'individus subit des mutations aléatoires de leurs mouvements.

### 6. Exploration et phéromones
Les chemins explorés sont enregistrés, et les impasses peuvent être fermées (système de phéromones).

## Sorties générées

### Fichiers de visualisation
- `tests/genetique/init.png` : Labyrinthe initial
- `tests/genetique/final.png` : Labyrinthe final avec modifications
- `tests/genetique/exploredFull.png` : Carte d'exploration complète
- `tests/genetique/param/c1.png` : Courbes d'évolution (min, max, moyenne)
- `tests/genetique/param/loss.png` : Courbe de loss (score minimum)
- `logs/explorations/{generation}.png` : Visualisations par phase

### Logs
Le fichier `logs/logs.txt` contient pour chaque phase :
- Taille du chemin réalisé vs théorique
- Statistiques des pénalités
- Score total
- Chemin emprunté compressé

## Personnalisation avancée

### Modifier les pénalités

```python
Individue.PENALITES = {
    "sortie de terrain": 1,
    "foncer dans un mur": 1,
    "bonus d'arrivee": 0,
    "position final": 3,        # Puissance pour la distance finale
    "retour en arriere": 1,
}
```

### Fréquence de génération des images

```python
# Dans simulation()
score_min, score_max, score_avg, dist = population.simulation(
    picture_each_x_generation=1000  # Génère une image toutes les 1000 générations
)
```

## Performances

Le projet inclut un système de profilage pour analyser les performances :

```python
python tests_genetique.py
# Affiche les 40 fonctions les plus coûteuses en temps d'exécution
```

## Exemple d'utilisation programmatique

```python
from MazeMaker import Maze, MazeColor
from Genetique import Population, Individue

# Configuration
maze = Maze(30, seed=123)
Population.set_generation(1000)
Population.set_individues(100)
Population.set_mutation(0.8)

# Simulation
population = Population(maze)
score_min, score_max, score_avg, dist = population.simulation(
    picture_each_x_generation=100
)

# Récupération du meilleur individu
best_individual = population.individues[0]
best_path = best_individual.get_path(maze)
print(f"Meilleur score: {best_individual.score}")
print(f"Longueur du chemin: {len(best_path)}")
```

## Détails techniques

### Encodage des mouvements
Les individus encodent leur chemin avec des entiers de 0 à 7 représentant 8 directions :

```
(-1, 1)  (0, 1)  (1, 1)
   1       0       7

(-1, 0)    X    (1, 0)
   2             6

(-1,-1)  (0,-1)  (1,-1)
   3       4       5
```

### Fonction de fitness
Le score est calculé selon la formule :

```
score = distance_finale³ + 
        sorties_terrain × 1 + 
        murs_heurtés × 1 + 
        retours_arrière × 1 + 
        bonus_arrivée × 0
```

Un score plus faible est meilleur.

## Limitations connues

- Les labyrinthes très grands (>500×500) peuvent nécessiter beaucoup de mémoire
- Le temps de calcul augmente avec la taille de la population et le nombre de générations
- Les visualisations avec `show_values=True` deviennent illisibles pour des labyrinthes >50×50

## Contributions et améliorations possibles

- Implémentation d'autres algorithmes de résolution (A*, colonies de fourmis)
- Optimisation des performances avec numba ou cython
- Interface graphique interactive
- Export des solutions au format vidéo
- Paramètres adaptatifs (mutation, sélection)

## Auteur et licence

Ce projet a été développé comme démonstration d'un algorithme génétique appliqué à la résolution de labyrinthes.

---

**Note** : Les visualisations utilisent matplotlib et peuvent nécessiter un backend graphique approprié selon votre environnement d'exécution.