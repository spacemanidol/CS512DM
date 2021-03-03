from sklearn import svm
import os
import argparse
import sys
import numpy as np
import pandas as pd
import numpy as np

def load_data(filename):
    x,y = [],[]
    with open(filename) as f:
        for line in f:
            line = line.strip().split(',')
            x.append(line[:-1])
            y.append(line[-1])
    return x, y

def get_f1(clf, xv, yv):
    tp, fp, tn,fn = 0,0,0,0
    for i, e in enumerate(xv):
        result = clf.predict([e])[0]
        if result == '0':
            if yv[i] == '0':
                tn += 1
            else:
                fn += 1
        else:
            if yv[i] =='0':
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
    data = pd.concat([df1, df2], axis=1)
    data0 = data[data.label == 0]
    data1 = data[data.label == 1]
    n = len(label)
    n1 = sum(label)
    n0 = n - n1
    l = []
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
        l.append(m_fisher_score)
    return l
def main(args):
    xt, yt = load_data(args.train_file)
    xv, yv = load_data(args.test_file)
    clf = svm.SVC()
    clf = clf.fit(xt, yt)
    print("Performance on Train:{}".format(clf.score(xt, yt)))
    print("Performance on Test:{}".format(clf.score(xv,yv)))
    print("F1 score:{}".format(get_f1(clf, xv, yv)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apriori on grocery data')
    parser.add_argument('--train_file', default='HW1_dataset/audit_risk/train.csv', type=str, help='input train file')
    parser.add_argument('--test_file',default='HW1_dataset/audit_risk/test.csv', type=str, help='input test file')
    args = parser.parse_args()
    main(args)    