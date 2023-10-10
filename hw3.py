#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

def parse_hw3_1(filename):
    data = []
    with open(filename) as file:
        for line in file.readlines():
            entry = [x for x in line.split()]
            data.append({'x': [float(entry[0]), float(entry[1])], \
                         'y': int(entry[2])})
    return data

def dist2(p1, p2):
    return math.sqrt(math.pow(p2[0]-p1[0],2.0)+math.pow(p2[1]-p1[1],2.0))

def one_NN(pt, data):
    best = data[0]
    for d in data:
        if dist2(pt,d['x']) < dist2(pt,best['x']):
            best = d
    return best['y']

def p1():
    data = parse_hw3_1('hw3Data/D2z.txt')
    x0s, x1s = np.mgrid[-2:2.05:0.1, -2:2.05:0.1]
    pts = np.mgrid[-2:2.05:0.1, -2:2.05:0.1].reshape(2,-1).T
    decisions = map(lambda pt: one_NN(pt,data), pts)
    colors = ['#ff8888' if y == 0 else '#8888ff' for y in decisions]

    datax0s = [d['x'][0] for d in data]
    datax1s = [d['x'][1] for d in data]
    datacolors = ['#ff0000' if d['y'] == 0 else '#0000ff' for d in data]


    fig, ax = plt.subplots()
    ax.scatter(x0s, x1s, s=5, c=colors, marker='.')
    ax.scatter(datax0s, datax1s, s=8, c=datacolors, marker='x')

    ax.set(xlim=(-2.05,2.05),ylim=(-2.05,2.05))

    plt.show()

def parse_hw3_2(filename):
    data = np.genfromtxt(open(filename), delimiter=',', encoding='utf8', dtype=str)
    words = data[0][1:]
    emails = np.array([ [int(x) for x in entry[1:3001]] for entry in data[1:]])
    answers = np.array([ int(entry[3001]) for entry in data[1:]])
    return words,emails,answers

def p2():
    words, emails, answers = parse_hw3_2('hw3Data/emails.csv')

    for iter in range(5):
        test = emails[1000*iter:1000*(iter+1)]
        training = [emails[i] for i in range(5000) if i not in range(1000*iter,1000*(iter+1))]
        ans = answers[1000*iter:1000*(iter+1)]
        training_ans = [answers[i] for i in range(5000) if i not in range(1000*iter,1000*(iter+1))]

        predicted = np.zeros(1000)
        for i in range(len(test)):
            if i% 100 == 0:
                print(str(i)+" iterations")
            d = test[i]
            best = 0
            for j in range(len(training)):
                d1 = training[j]
                if np.sum(np.abs(d-d1)) < np.sum(np.abs(d-training[best])):
                    best = j
            predicted[i] = training_ans[best]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(1000):
            if predicted[i] == 0 and ans[i] == 0:
                tn += 1
            elif predicted[i] == 1 and ans[i] == 0:
                fp += 1
            elif predicted[i] == 0 and ans[i] == 1:
                fn += 1
            elif predicted[i] == 1 and ans[i] == 1:
                tp += 1
        print("True positives: "+str(tp))
        print("True negatives: "+str(tn))
        print("False positives: "+str(fp))
        print("False negatives: "+str(fn))
        print("Accuracy: "+str((tp+tn)/1000))
        print("Precision: "+str((tp/(tp+fp))))
        print("Recall: "+str((tp/(tp+fn))))

def p3():
    words, emails, answers = parse_hw3_2('hw3Data/emails.csv')
    
    learnrate = 0.1
    steps = 10

    for iter in range(5):
        test = emails[1000*iter:1000*(iter+1)]
        training = [emails[i] for i in range(5000) if i not in range(1000*iter,1000*(iter+1))]
        ans = answers[1000*iter:1000*(iter+1)]
        training_ans = [answers[i] for i in range(5000) if i not in range(1000*iter,1000*(iter+1))]

        theta = np.array(1000)
        
