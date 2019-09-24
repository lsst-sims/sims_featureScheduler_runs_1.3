import numpy as np
from scipy.spatial import cKDTree as kdTree


np.random.seed(42)
# Make a square of random points
npts = int(1e7)
x = np.random.random(npts) - 0.5
y = np.random.random(npts) - 0.5
data = list(zip(x, y))
tree = kdTree(data)

count = np.size(tree.query_ball_point([0, 0], 0.2))
print('count = %i' % count)
