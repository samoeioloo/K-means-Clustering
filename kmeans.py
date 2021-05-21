# K means clustering
# Samoei Oloo
# 21 May 2021


import pandas as pd
import numpy as np
import random
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
        return np.linalg.norm(p1-p2, axis=0)

    # assign each data point to a centroid (minimum distance)
    def closest_centroid(allCentroids, point):
        new_centroid = []
        for i in point:
            dist = []
            for j in allCentroids:
                dist.append(distance(i,j))
            new_centroid.append(np.argmin(dist)) # find new centroid by selecting closest (least distance) centroid
        return new_centroid
        
    # assign first k points from dataset as initial centroids (examples 1,4,7 have been moved to be the first 3 points in dataset)
    def fit_data(self, data):
        self.centroids = {}
        #loop through and assign centroids (step 1)
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
                # Done by finding minimum distance (ie closest centroid) (step 2)

                cluster_num = distances.index(min(distances))
                self.classifications[cluster_num].append(pt) # add the point to its new cluster (step 3)

                 # Repeat above steps for all classifications in clusters
                for cluster_num in self.classifications:
                    self.centroids[cluster_num] = np.average(self.classifications[cluster_num], axis=0) # get new centroid
def main():
    # cluster datasets
    K = 3

    #set three centers as (2,10), (5,8), (1,2)
    #centroid1 = np.array([2,10])
    #centroid2 = np.array([5,8])
    #centroid3 = np.array([1,2])
    # read in sample data into a dataframe
    df = pd.read_excel('sample_dataset.xlsx', header=None)
    initial_centroids = [0,3,6] # indices of examples 1,4,7
    centroids = []
    for i in initial_centroids:
        centroids.append(df.loc[i]) # get data points of initial centroids

    centroids = np.array(centroids) # convert to 2d array
    print(centroids)
    # place data into clusters
    # cluster1 = np.random.randn(10,2) + centroid1
    # cluster2 = np.random.randn(10,2) + centroid2
    # cluster3 = np.random.randn(10,2) + centroid3
    #
    # data =  np.concatenate((cluster1, cluster2, cluster3), axis=0)
    #
    # # fit the data into the algo
    # kmeans = K_Means(K)
    # kmeans.fit_data(data)

    #print centroids
    #print(kmeans.centroids)


if __name__ == "__main__":
    main()
