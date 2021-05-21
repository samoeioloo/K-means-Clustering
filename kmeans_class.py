import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math

class K_Means:
    # initialise default k value, error tolerance and maximum iterations
    def __init__(self, k=3, tolerance=0.001, max_iterations = 500):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # calculate euclidean distance between to points
    def distance(self, p1, p2):
        return np.linalg.norm(point1-point2, axis=0)

    # assign first k points from dataset as initial centroids (examples 1,4,7 have been moved to be the first 3 points in dataset)
