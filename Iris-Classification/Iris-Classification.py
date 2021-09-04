# -*- coding:utf-8 -*-
import numpy as np 
import matplotlib.colors as colors
from sklearn import svm 
import sklearn.svm
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt 
import matplotlib as mpl

# Iris can be divided into three different varieties according to its calyx and petal size.
# data in CSV: sepal length, sepal width, petal length, petal width, type (target)


# load data 
# iris_type: turn string (bytes) into integer
def iris_type(s):
    index = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return index[s]

data_path = './iris.csv'
# data_path = '/home/aistudio/work/iris.csv'
data = np.loadtxt(
    data_path, # file path 
    dtype=float, # data type 
    delimiter=',', # CSV
    converters={4:iris_type}, # convert the 5th column into integer
)
print(data)
print(data.shape)
# data split 
x, y = np.split(
    data, 
    (4,), # columns [0-3]: X-axis, column 4: Y-axis
    axis=1, # split by column 
)
x = x[:, 0:2] # row-X: column [0-3], column-X: column [0-1] as feature for building graph 
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, # sample feature set 
    y, # sample result 
    random_state=1, # seed 
    test_size=0.3, # proportion of test samples
)


# build model 
def classifier():
   clf = svm.SVC(
       C=0.5, # error penalty coefficient, 1 by default 
       kernel="linear", 
       decision_function_shape='ovr',
   )
   return clf 

# definition of SVM model 
clf = classifier()


# train 
def train(clf, x_train, y_train):
    clf.fit(
        x_train, # feature vector of training set 
        y_train.ravel(), # target of training set 
)

# train SVM model 
train(clf=clf, x_train=x_train, y_train=y_train)


# evaluate 
def show_accuracy(a, b, tip):
    # check if a equals b
    # calculate average value of acc
    acc = (a.ravel() == b.ravel())
    print("%s accuracy: %.3f "%(tip, np.mean(acc)))

def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print("training prediction: %.3f "%(clf.score(x_train, y_train)))
    print("testing data prediction: %.3f "%(clf.score(x_test, y_test)))
    # compare original results and predicted results
    show_accuracy(clf.predict(x_train), y_train, "training data")
    show_accuracy(clf.predict(x_test), y_test, "testing data")
    # calculate decision function 
    # the three values represent the distance of x to each split plane
    print("decision_function:\n", clf.decision_function(x_train))

print_accuracy(clf, x_train, y_train, x_test, y_test)


# implement model 
def draw(clf, x):
    iris_feature = ['sepal length', "sepal width", 'petal length', "petal width"]
    x0_min, x0_max = x[:, 0].min(), x[:, 0].max() # range of column 0
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max() # range of column 1
    x0, x1 = np.mgrid[x0_min:x0_max:200j, x1_min:x1_max:200j] # generate grid sampling points 
    grid_test = np.stack((x0.flat, x1.flat), axis=1)
    print("grid_test: \n", grid_test)
    # print the distance from the sample to the decision plane 
    print("the distance to decision plane: \n", clf.decision_function(grid_test))

    grid_hat = clf.predict(grid_test) # predicted categorical value
    print("grid_hat: \n", grid_hat)
    grid_hat = grid_hat.reshape(x0.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', "#FFA0A0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    plt.pcolormesh(x0, x1, grid_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark) # sample point 
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)       # test point 
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)
    plt.title('SVM in iris data classification', fontsize=30)
    plt.grid()
    plt.show()


draw(clf, x)