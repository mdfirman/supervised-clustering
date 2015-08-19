import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import squareform


class CorrClust(object):
    '''
    class like an sklearn class, which can be trained to do correlation
    clustering on a new dataset
    '''

    def __init__(self, balance_training=False, forest_params={},
            num_random_starts=10, max_iters=100):
        self.forest_params = forest_params
        self.balance_training = balance_training
        self.num_random_starts = num_random_starts
        self.max_iters = max_iters

    def train(self, X, Y, subsample_length):
        '''
        training the model
        unsure if parameters should go here or in the initialisation function
        '''
        if self.balance_training:
            x_pairs, y_pairs = self._form_balanced_pairs(X, Y, subsample_length)
        else:
            x_pairs, y_pairs = self._form_pairs(X, Y, subsample_length)

        print "Training - there are %d +ve pairs and %d -ve ones" % \
            ((y_pairs==0).sum(), (y_pairs==1).sum())

        self.rf = RandomForestClassifier(**self.forest_params)
        self.rf.fit(x_pairs, y_pairs)

    def test(self, X):
        '''
        running the model on test data
        TODO - allow for just a subset of edges to be formed, thus creating
        a sparse matrix of edge probabilities
        '''
        x_pairwise = self._form_pairs(X)
        edge_probabilities = self.rf.predict_proba(x_pairwise)[:, 1]
        prob_matrix = squareform(edge_probabilities)
        y_prediction = self._correlation_clusterer(prob_matrix)
        return y_prediction

    def _form_pairs(self, X, Y=None, subsample_length=None):

        # here I shall be taking all pairs from the data
        idxs1, idxs2 = self._pair_idxs(X.shape[0])

        if (subsample_length is not None) and (subsample_length < idxs1.shape[0]):
            print subsample_length, idxs1.shape[0]
            to_use = np.random.choice(idxs1.shape[0], subsample_length, replace=False)
            idxs1 = idxs1[to_use]
            idxs2 = idxs2[to_use]

        x_pairs = np.abs(X[idxs1] - X[idxs2])

        if Y is not None:
            return x_pairs, Y[idxs1] == Y[idxs2]
        else:
            return x_pairs

    def _form_balanced_pairs(self, X, Y, subsample_length=None):
        '''
        forming pairs with equal +ve and -ve edges
        must be given Y vector for this to work
        '''
        idxs1, idxs2 = self._pair_idxs(X.shape[0])
        classes = Y[idxs1] == Y[idxs2]

        # working out how many edges I can use in total
        max_edges = np.array(classes.sum(), (1-classes).sum()).min()

        if subsample_length is not None:
            max_edges = min(max_edges, subsample_length/2)

        # subsample each class in turn
        to_use = np.hstack([np.random.choice(
                np.where(classes==this_class)[0], max_edges, replace=False)
                for this_class in [0, 1]])

        # print final_idxs1, final_idxs2
        x_pairs = np.abs(X[idxs1[to_use]] - X[idxs2[to_use]])
        y_pairs = classes[to_use]
        return x_pairs, y_pairs

    def _pair_idxs(self, num_data):

        A = np.outer(np.arange(num_data), np.ones(num_data))
        idxs1 = squareform(A, force='to_vector', checks=False)
        idxs2 = squareform(A.T, force='to_vector', checks=False)
        return idxs1.astype(int), idxs2.astype(int)

    def _correlation_clusterer(self, edge_probabilities):
        '''
        does the actual coorelation clustering, given edge probabilities
        edge_probabilities can be a sparse matrix.
        '''

        # convert edge probabilities to weights
        edge_probabilities[edge_probabilities==0] = 0.0001
        edge_probabilities[edge_probabilities==1] = 1.0 - 0.0001
        weights = np.log(edge_probabilities / (1.0 - edge_probabilities))
        np.fill_diagonal(weights, 0)

        self.weights = weights

        # form some different starting guesses
        # use: all in same cluster, all in own cluster, then random clusterings
        N = weights.shape[0]
        start_points = [np.ones(N), np.arange(N)] + \
            [np.random.randint(0, N, N) for _ in range(self.num_random_starts)]

        # setting variables to keep track in the loop
        max_energy = -np.inf
        best_Y = None

        # for each starting point, run the solver
        for start_point in start_points:
            Y, energy = self._clustering_solver(weights, start_point.astype(int))
            if energy > max_energy:
                max_energy = energy
                best_Y = Y

        return best_Y, max_energy

    def _clustering_solver(self, W, start_labels):
        '''
        the actual code that does the clustering, using the AL_ICM algorithm
        trying to MAXIMISE the energy

        This could probably be done in a much more efficient way, without the
        need for bincount on each inner loop
        '''
        iteration = 0
        labels = start_labels.copy()
        n_items = W.shape[0]
        old_energy = -np.inf

        for iteration in range(self.max_iters):

            # assign each item in turn to the best cluster
            for j in range(n_items):

                cluster_scores = np.bincount(labels, weights=W[j, :])

                if np.all(cluster_scores < 0):
                    # creating a new label
                    labels[j]  = labels.max() + 1
                else:
                    # assigning to the best exisiting label
                    labels[j]  = np.argmax(cluster_scores)

            if iteration % 15 == 0:
                # reasign labels for efficiency
                _, labels = np.unique(labels, return_inverse=True)

            energy = self._clustering_energy(W, labels)

            if energy < old_energy:
                raise Exception("This should never happen!")
            elif energy == old_energy:
                break

            old_energy = energy

        else:
            print "Reached max iters (%d), breaking" % iteration

        _, labels = np.unique(labels, return_inverse=True)
        return labels, energy

    def _clustering_energy(self, W, Y):
        '''
        sums up all the edges between items which have been given the same
        class label
        '''
        Y = Y.copy()[None, :]
        return (W * (Y==Y.T).astype(float)).sum()
