import math
import numpy as np
import matplotlib.pyplot as plt


def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will repeat one time more than needed.

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def vector(point1, point2):
    point3 = []
    if len(point1) == len(point2):
        for i in range(0, len(point1)):
            point3.append(point2[i] - point1[i])
    return point3

def module(point1,point2):
    sum=0
    if len(point1) == len(point2):
        point3=vector(point1,point2)
        for i in range(0, len(point3)):
            sum = sum + point3[i]*point3[i]
        return math.sqrt(sum)

def scalarProduct(vector1,vector2):
    sum = 0
    if(len(vector1)==len(vector2)):
        for i in range(0, len(vector1)):
            sum = sum + vector1[i]*vector2[i]
        return sum
    else:
        print("ERROR")
        return 0

def pointsEqual(point1,point2):
    if (len(point1)==len(point2)):
        comparison = point1 == point2
        equal_arrays = comparison.all()
        return equal_arrays
    else:
        return False

def ABOF(point, tab):
    ABOF=0
    module1=0.
    module2=0.
    sProduct1=0.
    lot1=0.
    for i in range(0,len(tab)):
        if pointsEqual(point,tab[i])==False:
            for j in range(i,len(tab)):
                if pointsEqual(point,tab[j])==False: #and pointsEqual(tab[j],tab[i])==False

                    module1=module1+1.0/(module(point,tab[i])*module(point,tab[j]))
                    module2 = module2 + 1.0 / (module(point, tab[i])*module(point, tab[i]) * module(point,tab[j])*module(point,tab[j]))
                    sProduct1=sProduct1+scalarProduct(vector(point,tab[i]),vector(point,tab[j]))
                    lot1=lot1+scalarProduct(vector(point,tab[i]),vector(point,tab[j]))/math.pow(module(point, tab[i])*module(point, tab[i]) * module(point,tab[j])*module(point,tab[j]),2)
    ABOF = module1*lot1
    ABOF=ABOF/module1
    ABOF=ABOF-math.pow(((module1*sProduct1*module2)/module1),2)
    return ABOF

params = [[[ 0,1],  [ 0,1]],
          [[ 5,1],  [ 5,1]],
          [[-2,5],  [ 2,5]],
          [[ 2,1],  [ 2,1]],
          [[-5,1],  [-5,1]]]

n = 20
dims = len(params[0])

data = []
y = []
for ix, i in enumerate(params):
    inst = np.random.randn(n, dims)
    for dim in range(dims):
        inst[:,dim] = params[ix][dim][0]+params[ix][dim][1]*inst[:,dim]
        label = ix + np.zeros(n)

    if len(data) == 0: data = inst
    else: data = np.append( data, inst, axis= 0)
    if len(y) == 0: y = label
    else: y = np.append(y, label)

num_clusters = len(params)

resultsABOF=[]
tab=[]
for i in data:
    resultsABOF.append(abs(ABOF(i,data))/1E+14)
    # print(i)
    # print(abs(ABOF(i,data))/1E+15)
for i in range(0,len(resultsABOF)):
    # print(resultsABOF[i])
    if(resultsABOF[i]<0.5): #parametre a changer pour le degré de liberte
        tab.append(data[i])

x, y = zip(*data)
z,w=zip(*tab)
plt.scatter(x, y,color="blue")
plt.scatter(z, w,color="red")
plt.show()



