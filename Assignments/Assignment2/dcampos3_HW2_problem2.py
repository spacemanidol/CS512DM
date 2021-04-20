import os
import argparse
import sys
import numpy as np
from mat4py import loadmat
from pyod.models.vae import VAE
from pyod.models.pca import PCA
from pyod.models.knn import KNN   # kNN detector
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from pyod.utils.utility import precision_n_scores

def print_metrics(y, y_pred):
    print("Precision @10:{}".format(np.round(precision_n_scores(y, y_pred, n=10),4)))
    print("Precision @20:{}".format(np.round(precision_n_scores(y, y_pred, n=20),4)))
    print("Precision @50:{}".format(np.round(precision_n_scores(y, y_pred, n=50),4)))
    print("Precision @100:{}".format(np.round(precision_n_scores(y, y_pred, n=100),4)))
    print("Precision @1000:{}".format(np.round(precision_n_scores(y, y_pred, n=1000),4)))
    print("ROC AUC Score:{}".format(np.round(roc_auc_score(y, y_pred),4)))
    print("Recall Score:{}".format(np.round(recall_score(y, y_pred),4)))

def main(args):
    data = loadmat(args.filename)
    trainx, testx, trainy, testy = train_test_split(data['X'], data['y'], test_size=args.train_split, random_state=2)
    valx, evalx, valy, evaly = train_test_split(testx, testy, test_size=0.5)
    data_size = len(trainx[0])
    encoder_neurons = [data_size, data_size/2, data_size/4]
    clf = KNN()
    clf.fit(trainx)
    print("Results Validation KNN")
    print_metrics(valy, clf.predict(valx))
    print("Results Evaluation KNN")
    print_metrics(evaly, clf.predict(evalx))

    clf = PCA(n_components=args.components)
    clf.fit(trainx)
    print("Results Validation PCA")
    print_metrics(valy, clf.predict(valx))
    print("Results Evaluation PCA")
    print_metrics(evaly, clf.predict(evalx))

    clf = VAE(encoder_neurons=encoder_neurons, decoder_neurons=encoder_neurons[::-1], epochs=args.epochs, contamination=args.contamination, gamma=args.gamma, capacity=args.capacity)
    clf.fit(trainx)
    print("Results Validation VAE")
    print_metrics(valy, clf.predict(valx))
    print("Results Evaluation VAE")
    print_metrics(evaly, clf.predict(evalx))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PYOD3')
    parser.add_argument('--components', default=3, type=int, help='Number of PCA components')
    parser.add_argument('--contamination', default=0.1, type=float, help='Percent of samples outliers')
    parser.add_argument('--capacity', default=0.2, type=float, help='VAE capacity')
    parser.add_argument('--gamma', default=0.8, type=float, help='Gamma value for VAE')
    parser.add_argument('--epochs', default=30, type=int, help='Epochs to train VAE')
    parser.add_argument('--method', default='KNN', choices=['KNN','PCA', 'VAE'], type=str, help='Outlier detection method')
    parser.add_argument('--train_split', default=0.6, type=float, help='Percent of samples going to train')
    parser.add_argument('--filename', default='CS512_HW2_dataset/vowels.mat', type=str, help='MAT file to run Outlier Detection on')
    args = parser.parse_args()
    main(args)    