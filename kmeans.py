# K means clustering
# Samoei Oloo
# 21 May 2021


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# calculate euclidean distance between to points
def distance(p1, p2):
    return np.linalg.norm(p1-p2, axis=0)

# returns index of new assigned centroid
def closest_centroid(allCentroids, points):
    new_centroid = []
    for i in points:
        dist = []
        for j in allCentroids:
            dist.append(distance(i,j))
        new_centroid.append(np.argmin(dist)) # find new centroid by selecting closest (least distance) centroid
    return new_centroid

# computes new centroids and assigns
def calc_centroids(clusters, points):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(points), pd.DataFrame(clusters, columns=['Cluster'])], axis = 1) #create new dataframe with the new clusters

    for c in set(new_df['Cluster']):
        cluster = new_df[new_df['Cluster'] == c][new_df.columns[:-1]]
        # find mean value of all points assigned to the centroid
        cluster_avg = cluster.mean(axis=0)
        new_centroids.append(cluster_avg) # move the centroid number to its average

    return new_centroids

# printing cluster info
def print_clusters(new_centroids_param):
    cluster1 = []
    cluster2 = []
    cluster3 = []

    for i in range(len(new_centroids_param)):
        if (new_centroids_param[i] == 0): #cluster1
            cluster1.append(i+1)
        elif (new_centroids_param[i] == 1): #cluster2
            cluster2.append(i+1)
        elif (new_centroids_param[i] == 2): #cluster3
            cluster3.append(i+1)
        else:
            print("Error. Check indexing of get_new_centroids")

    print("Cluster 1 :")
    print(",".join([str(i) for i in cluster1]))
    print("\nCluster 2 :")
    print(",".join([str(i) for i in cluster2]))
    print("\nCluster 3 :")
    print(",".join([str(i) for i in cluster3]))
    return ""

def main():
    # cluster datasets
    K = 3
    num_iters = 0;
    iter = "\t\tIteration "

    df = pd.read_excel('sample_dataset.xlsx', header=None)
    initial_centroids = [0,3,6] # indices of examples 1,4,7
    all_points = []
    centroids = []

    for i in initial_centroids:
        centroids.append(df.loc[i]) # get data points of initial centroids

    for i in range(df.shape[0]):
        all_points.append(df.loc[i])

    centroids = np.array(centroids) # convert to 2d array
    all_points = np.array(all_points) # convert to 2d array


    print(iter + str(num_iters) + "\n")
    # print clusters for inital iteration
    # not invoking the calc_centroids because we have already been assigned the centroids in the spec
    print(print_clusters(closest_centroid(centroids, all_points)))

    #print centroids for initial iteration
    print("Centroids: ")
    print(centroids.tolist())
    print()
    # NEXT:
    num_iters += 1
    print(iter + str(num_iters))
    get_new_centroids = closest_centroid(centroids, all_points)
    centroids = calc_centroids(get_new_centroids, all_points)
    print(print_clusters(closest_centroid(centroids, all_points)))
    print("Centroids: ")
    print(np.array(centroids).tolist())
    print()

    the_new_centroids = []

    while ( (the_new_centroids == get_new_centroids) is not True): # while they arent equal
        num_iters += 1
        print(iter + str(num_iters))
        get_new_centroids = closest_centroid(centroids, all_points)
        centroids = calc_centroids(get_new_centroids, all_points)

        print(print_clusters(closest_centroid(centroids, all_points)))
        print("Centroids: ")
        print(np.array(centroids).tolist())
        print()
        the_new_centroids = closest_centroid(centroids, all_points)
    print("The newly computed centroids are the same as the previous ones.\nConvergence criterion met. \nFinal iteration number : " + str(num_iters) + "\nClustering complete.")


    plt.figure()
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
    plt.scatter(all_points[:, 0], all_points[:, 1], alpha=0.1)
    plt.show()

if __name__ == "__main__":
    main()
