import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Génération de données aléatoires
np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
c = np.random.rand(100)  # Quatrième dimension

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Représentation des données avec la couleur comme quatrième dimension
img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()