import os
import numpy as np
import cv2
import random
import mahotas
from mahotas.features import surf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
import misvm

def extract_features(image, NUM_BINS):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [NUM_BINS, NUM_BINS, NUM_BINS], [0, 256, 0, 256, 0, 256]).flatten()
    hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    lbp = mahotas.features.lbp(gray, 1, 8)
    tas = mahotas.features.tas(gray)
    zern = mahotas.features.zernike_moments(gray, 28)
    return np.hstack([hist, haralick, hu_moments, lbp, tas, zern])

def load_data(train_test_ratio, NUM_BINS=4, DATA_SIZE=999999999):
    normal = []
    mutated = []
    count = 0
    for f in os.listdir('tcga_coad_msi_mss_jpg/MSS_JPEG'):
        img = cv2.imread('tcga_coad_msi_mss_jpg/MSS_JPEG/' + f, 1)
        normal.append(extract_features(img, NUM_BINS))
        count += 1
        if (count%10000==0):
            print(count)
        if (count > DATA_SIZE):
            break
    count = 0
    print()
    for g in os.listdir('tcga_coad_msi_mss_jpg/MSIMUT_JPEG'):
        img = cv2.imread('tcga_coad_msi_mss_jpg/MSIMUT_JPEG/' + g, 1)
        mutated.append(extract_features(img, NUM_BINS))
        count += 1
        if (count%7500==0):
            print(count)
        if (count > DATA_SIZE):
            break
    print()
    data = StandardScaler().fit_transform(normal+mutated)
    labels = [0 for n in normal]
    labels = labels + [1 for m in mutated]
    labels = np.array(labels)
    idx = np.random.permutation(len(data))
    X, y = data[idx], labels[idx]
    X = PCA(n_components=110).fit_transform(X)
    cutoff = int(len(X)*train_test_ratio)
    return X[:cutoff], y[:cutoff], X[cutoff:], y[cutoff:]

def build_svm(train_x, train_y, test_x, test_y):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FailedFitWarning)
    #Tuning
    #parameters = {'C': [1+0.1*x for x in range(0, 10)], 'coef0': [1+0.1*x for x in range(0, 10)], 'degree': [x for x in range(3, 7)], 'kernel':('poly', 'rbf', 'sigmoid')}
    #clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=10, n_jobs=-1, scoring='accuracy')

    clf = SVC(kernel='poly', degree=4, shrinking=True, C=1.0, coef0=1.2, max_iter=100000, verbose=2)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    correct = 0
    for i in range(0, len(test_y)):
        if (predictions[i] == test_y[i]):
            correct+=1
    print('Accuracy: %f' %(correct/len(test_y)))
    return clf

def plot_hist_size():
    accs = []
    for h in range(4, 20):
        train_x, train_y, test_x, test_y = load_data(0.8, NUM_BINS=h, DATA_SIZE=5000)
        accuracy = 0
        batch_train = len(train_x)//10
        batch_test = len(test_x)//10
        for b in range(0, 10):
            accuracy += build_svm(train_x[batch_train*b:batch_train*(b+1)], train_y[batch_train*b:batch_train*(b+1)], test_x[batch_test*b:batch_test*(b+1)], test_y[batch_test*b:batch_test*(b+1)])
        accs.append(accuracy/10)
        print(h)
    return accs
