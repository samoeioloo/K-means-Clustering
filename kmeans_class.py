# K means clustering
# Samoei Oloo
# 21 May 2021


import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math

class K_Means:
    # initialise default k value, error tolerance and maximum iterations
    def __init__(self, k=3, tolerance=0.001, max_iters = 500):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    # calculate euclidean distance between to points
    def distance(self, p1, p2):
        return np.linalg.norm(point1-point2, axis=0)

    # assign first k points from dataset as initial centroids (examples 1,4,7 have been moved to be the first 3 points in dataset)
    def fit_data(self, data):
        self.centroids = {}
        #loop through and assign centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        # begin clustering
        for i in range(self.max_iters):
            # create k classifications
            self.classifications = {}
            for j in range(self.k):
                self.classifications[j] = [] #clear

            # calc distance between points and centroids
            for pt in data:
                distances = [] #list of distances for each point
                for cntrd in self.centroids: # loop through centroids
                    distances.append(self.distance(pt,self.centroids[cntrd]))

                # Compute the cluster each point belongs to
                # Done by finding minimum distance

                cluster_num = distances.index(min(distances))
                self.classifications[cluster_num].append(pt) # add the point to its new cluster

                 # Repeat above steps for all classifications in clusters
                for cluster_num in self.classifications:
                    self.centroids[cluster_num] = np.average(self.classifications[cluster_num], axis=0) # get new centroid
