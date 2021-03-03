from sklearn import svm
import os
import argparse
import sys
import numpy as np
import pandas as pd
import numpy as np

def load_data(filename, labeled):
    x,y = [],[]
    with open(filename) as f:
        for line in f:
            line = line.strip().split(',')
            if labeled:
                x.append([float(a) for a in line[:-1]])
                y.append(int(line[-1]))
            else:
                x.append([float(a) for a in line])
    return x, y

def get_f1(clf, xv, yv):
    tp, fp, tn,fn = 0,0,0,0
    for i, e in enumerate(xv):
        result = clf.predict([e])[0]
        if result == 0:
            if yv[i] == 0:
                tn += 1
            else:
                fn += 1
        else:
            if yv[i] == 0:
                fp += 1
            else:
                tp += 1 
    print("True Positive:{}\nFalse Positive:{}\nTrue Negative:{}\nFalse Negative:{}".format(tp, fp, tn, fn))
    print("Precision:{}".format(tp/(tp+fp)))
    print("Recall:{}".format(tp/(tp+fn)))            
    return tp/(tp + 0.5*(fp+fn))

def get_fisher_score(x, y):
    df1 = pd.DataFrame(x)
    df2 = pd.DataFrame(y)
    df2.rename(columns = {0:'label'}, inplace = True) 
    data = pd.concat([df1, df2], axis=1)
    data0 = data[data.label == 0]
    data1 = data[data.label == 1]
    n = len(y)
    n1 = sum(y)
    n0 = n - n1
    fisher_scores = []
    features_list = list(data.columns)[:-1]
    for feature in features_list:
        m0_feature_mean = data0[feature].mean() 
        m0_SW=sum((data0[feature] -m0_feature_mean )**2)
        m1_feature_mean = data1[feature].mean()
        m1_SW=sum((data1[feature] -m1_feature_mean )**2)
        m_all_feature_mean = data[feature].mean() 
        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2
        m_SB = m1_SB + m0_SB
        m_SW = (m0_SW + m1_SW) / n
        if m_SW == 0:
            m_fisher_score = 0
        else:
            m_fisher_score = m_SB / m_SW
        fisher_scores.append(m_fisher_score)
    return fisher_scores

def optimize_features(fisher_scores, threshold, x):
    selected_idx = [i for (i,j) in enumerate(fisher_scores) if j>=threshold]
    print("Using the following indexes in training data:{}\nAll fisher scores as follows{}".format(selected_idx, fisher_scores))
    new_x = []
    for i, j in enumerate(x):
        new_x.append([j[k] for k in selected_idx])
    return new_x 

def main(args):
    xt, yt = load_data(args.train_file, True)
    xv, yv = load_data(args.test_file, True)
    clf = svm.SVC(probability=True)
    clf = clf.fit(xt, yt)
    print("Performance on Train:{}".format(clf.score(xt, yt)))
    print("Performance on Test:{}".format(clf.score(xv,yv)))
    print("F1 score:{}".format(get_f1(clf, xv, yv)))
    ux, _ = load_data(args.augment_file, False) # unlabelled augmentable data
    if args.do_fisher_score:
        fisher_scores = get_fisher_score(xt, yt)
        xt = optimize_features(fisher_scores, args.fisher_threshold, xt)
        xv = optimize_features(fisher_scores, args.fisher_threshold, xv)
        print("###########################################")
        print("Performance using fisher scores and a threshold of {}".format(args.fisher_threshold))
        clf = svm.SVC(probability=True)
        clf = clf.fit(xt, yt)
        print("Performance on Train:{}".format(clf.score(xt, yt)))
        print("Performance on Test:{}".format(clf.score(xv,yv)))
        print("F1 score:{}".format(get_f1(clf, xv, yv)))
        ux = optimize_features(fisher_scores, args.fisher_threshold, ux)
    stx, sty = [], [] # self train x and self train y
    for i, j in enumerate(ux):
        probs_0 =  clf.predict_proba([j])[0][0]
        probs_1 = 1 - probs_0
        if abs(probs_0-probs_1) > args.threshold:
            stx.append(j)
            if probs_1 > probs_0:
                sty.append(1)
            else:
                sty.append(0)
    print("Using original classifier we find {} samples have a least a {} confidence margin toward one label".format(len(sty), args.threshold))
    xt = xt + stx
    yt = yt + sty
    clf = svm.SVC(probability=True)
    clf = clf.fit(xt, yt)
    print("Performance on Train:{}".format(clf.score(xt, yt)))
    print("Performance on Test:{}".format(clf.score(xv,yv)))
    print("F1 score:{}".format(get_f1(clf, xv, yv)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apriori on grocery data')
    parser.add_argument('--train_file', default='HW1_dataset/audit_risk/train.csv', type=str, help='input train file')
    parser.add_argument('--test_file',default='HW1_dataset/audit_risk/test.csv', type=str, help='input test file')
    parser.add_argument('--augment_file', default='HW1_dataset/audit_risk/unlabelled.csv', type=str, help='input unlabelled file')
    parser.add_argument('--do_fisher_score', action='store_true', help='Do fisher score train data subseting')
    parser.add_argument('--fisher_threshold', default=1.0, type=float, help='Threshold for fisher score feature selection')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold for min confidence to keep data point and label')
    args = parser.parse_args()
    main(args)    