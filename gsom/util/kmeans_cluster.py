from sklearn.cluster import k_means
import numpy as np


class KMeansSOM:

    def _som_to_array(self, som_map):
        som_map_array = []
        for x in range(som_map.shape[0]):
            for y in range(som_map.shape[1]):
                som_map_array.append(som_map[x, y])
        return som_map_array

    def cluster_SOM(self, som_map, n_clusters=2):
        """
        Parameters
        ----------
        som_map : self organizing map
            2D array of weight vectors in SOM.
        n_clusters : number of clusters.

        Returns
        -------
        som_list : list
            list of the som nodes
        centroid : list
            cluster centroids.
        labels : list
            cluster label w.r.t. som node data-point as in som_list
        """

        som_list = self._som_to_array(som_map)

        clf = k_means(som_list, n_clusters=n_clusters)

        centroids = clf[0]
        labels = clf[1]

        return som_list, centroids, labels

    def _gsom_to_array(self, gsom_map):
        gsom_map_array = []
        for node in gsom_map.items():
            gsom_map_array.append(node[1].recurrent_weights)
        gsom_array = np.asarray(gsom_map_array).reshape((len(gsom_map_array)),len(gsom_map_array[0][0]))
        return gsom_array

    def cluster_GSOM(self, gsom_map, n_clusters=2):
        """
        Parameters
        ----------
        gsom_map : growing self organizing map
            2D array of weight vectors in SOM.
        n_clusters : number of clusters.

        Returns
        -------
        gsom_list : list
            list of the gsom nodes
        centroid : list
            cluster centroids.
        labels : list
            cluster label w.r.t. gsom node data-point as in gsom_list
        """

        gsom_list = self._gsom_to_array(gsom_map)

        clf = k_means(gsom_list, n_clusters=n_clusters)

        centroids = clf[0]
        labels = clf[1]
        clusters = self.seperate_clusters(labels,n_clusters)
        return clusters

    # seperate nodes into seperate clusters
    def seperate_clusters(self, nodes, number_of_clusters):
        clusters = {}
        for i in range (number_of_clusters):
            clusters[i] = []

        for val,cluster in enumerate(nodes):
            clusters[cluster].append(val)

        return clusters
