# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:48:38 2016

@author: mota2890
## restaurant review

"""
import numpy as np

lines = [line.strip() for line in open('restaurant_train.txt') if '\n' in line]
temp = [l.split(' ') for l in lines ]
print(temp[0])

print(len(temp[-1]))
dim = max(len(x) for x in temp)

f = np.zeros((1000,9490), dtype=int)
classes = [x[0] for x in temp ]
w = np.zeros((3,9490))

print(f[0][12])
y=12
f[0][11] =1

print(f[0][11])

maxi = 0
for i in range(0,len(temp)):
    for j in range(1,len(temp[i])):
        y= temp[i][j].split(":")
        if(maxi < int(y[0])):
            maxi = int(y[0])
            

for i in range(0,len(temp)):
    for j in range(1,len(temp[i])):
        y= temp[i][j].split(":")
        p=int(y[0])
        f[i][p]=1

traindata = zip(classes, f)
def classify(f, w):
    classscores = {c:(f.dot(w[c])) for c in range(len(w))}
    return max(classscores, key = classscores.get)


errors = 1
while errors > 0:
    errors = 0
    for correctclass, features in traindata:
        guessedclass = classify(features, w)
        if correctclass != guessedclass:
            w[correctclass] += features
            w[guessedclass] -= features
            errors += 1
    print("ERRORS:", errors)
    
print(classify(f[997],w))
  