from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

'''
This method calculates the soft assignments for each data point using GMM given the data and the
number of clusters. This method will construct the clusters and return the soft assignments for 
each data point.
'''
def make_clusters(input, k):
    gmm = GaussianMixture(n_components=k).fit(input)
    labels = gmm.predict(input)
    probabilities = gmm.predict_proba(input)

    return labels, probabilities

