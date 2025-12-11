# ...existing code...
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# ...existing code...

os.makedirs("images", exist_ok=True)

maze_perso = np.array(
    [
        [(255, 0, 0), (0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
        [(0, 0, 0), (255, 255, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
        [(0, 0, 0), (255, 255, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(255, 255, 255), (0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
    ],
    dtype=np.uint8
)

# sauvegarde exacte 5x5 pixels via PIL
Image.fromarray(maze_perso).save("images/given_maze5.png")

# (optionnel) affichage avec matplotlib sans redimensionnement
plt.figure(figsize=(2, 2))
plt.imshow(maze_perso, interpolation="nearest")
plt.axis("off")
plt.show()
# ...existing code...