#!/usr/bin/python
# Challenge two :Robot Execution Failures Test

import sys
import numpy as np
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
import ipdb

#svc = svm.SVC(kernel='poly', degree=3)
svc = svm.SVC(kernel='rbf', C=1, gamma='auto')
sgd = SGDClassifier(loss="hinge", penalty="l2")
clf = tree.DecisionTreeClassifier()
content = ['1,2','3,4']
dataNormalN = 0
dataCollisionN = 0
dataFrontCollisionN = 0
dataBackCollisionN = 0
dataLeftCollisionN = 0
dataRightCollisionN = 0
dataObstructionN = 0
dataMovedN = 0
dataSliMovedN = 0
dataLostN = 0
dataBottCollisionN = 0
dataBottObstructionN = 0
dataPartCollisionN = 0
dataToolCollisionN = 0

def readMeasure():
    global content
    if len(content) == 0:
        return None
    #ipdb.set_trace()
    tmp = ''
    if len(content) > 0:
        tmp = content.pop()
    arr = np.fromstring(tmp, sep='\t')
    for x in xrange(14):
        if len(content) > 0:
            tmp = content.pop()
        arr_ = np.fromstring(tmp, sep='\t')
        arr = np.r_[arr, arr_]
    return arr

def init(dbg = False):
    global content, dataNormalN, dataCollisionN, dataFrontCollisionN, \
    dataBackCollisionN, dataLeftCollisionN, dataRightCollisionN, \
    dataObstructionN, dataMovedN, dataSliMovedN, dataLostN, dataBottCollisionN, \
    dataBottObstructionN, dataPartCollisionN, dataToolCollisionN
    if dbg:
        print('PREPROCESSING DATA...')
        print('Loading lp1.data')
    
    with open("lp1.data") as f:
        content = f.readlines()
    content.reverse()
    if dbg:
        print('# of lines = ' + str(len(content)) + '; spected ' + str(len(content)/18.) + ' entries')
    tmp = '\n'
    #ipdb.set_trace()
    while len(content) > 0:
        tmp = content.pop()
        while tmp == '\n':
            if len(content) > 0:
                tmp = content.pop()
            else:
                break
        if tmp == 'normal\n':
            arr = readMeasure()
            if dataNormalN == 0:
                dataNormal = arr
                dataNormalN += 1
            else:
                dataNormal = np.vstack((dataNormal, arr))
                dataNormalN += 1

        elif tmp == 'collision\n':
            arr = readMeasure()
            if dataCollisionN == 0:
                dataCollision = arr
                dataCollisionN += 1
            else:
                dataCollision = np.vstack((dataCollision, arr))
                dataCollisionN += 1

        elif tmp == 'fr_collision\n':
            arr = readMeasure()
            if dataFrontCollisionN == 0:
                dataFrontCollision = arr
                dataFrontCollisionN += 1
            else:
                dataFrontCollision = np.vstack((dataFrontCollision, arr))
                dataFrontCollisionN += 1

        elif tmp == 'obstruction\n':
            arr = readMeasure()
            if dataObstructionN == 0:
                dataObstruction = arr
                dataObstructionN += 1
            else:
                dataObstruction = np.vstack((dataObstruction, arr))
                dataObstructionN += 1
    if dbg:
        print('Normal= ' + str(dataNormalN) + '; collision= ' + str(dataCollisionN) + \
            '; frontCollision= ' + str(dataFrontCollisionN) + '; obstruction= ' + str(dataObstructionN))
        print('Related entries = ' + str(dataNormalN + dataCollisionN + dataFrontCollisionN + dataObstructionN))

    if dbg:
        print('Loading lp2.data')
    with open("lp2.data") as f:
        content = f.readlines()
    content.reverse()
    if dbg:
        print('# of lines = ' + str(len(content)) + '; spected ' + str(len(content)/18.) + ' entries')
    tmp = '\n'
    while len(content) > 0:
        tmp = content.pop()
        while tmp == '\n':
            if len(content) > 0:
                tmp = content.pop()
            else:
                break;
        if tmp == 'normal\n':
            arr = readMeasure()
            if dataNormalN == 0:
                dataNormal = arr
                dataNormalN += 1
            else:
                dataNormal = np.vstack((dataNormal, arr))
                dataNormalN += 1

        elif tmp == 'front_col\n':
            arr = readMeasure()
            if dataFrontCollisionN == 0:
                dataFrontCollision = arr
                dataFrontCollisionN += 1
            else:
                dataFrontCollision = np.vstack((dataFrontCollision, arr))
                dataFrontCollisionN += 1

        elif tmp == 'back_col\n':
            arr = readMeasure()
            if dataBackCollisionN == 0:
                dataBackCollision = arr
                dataBackCollisionN += 1
            else:
                dataBackCollision = np.vstack((dataBackCollision, arr))
                dataBackCollisionN += 1

        elif tmp == 'right_col\n':
            arr = readMeasure()
            if dataRightCollisionN == 0:
                dataRightCollision = arr
                dataRightCollisionN += 1
            else:
                dataRightCollision = np.vstack((dataRightCollision, arr))
                dataRightCollisionN += 1

        elif tmp == 'left_col\n':
            arr = readMeasure()
            if dataLeftCollisionN == 0:
                dataLeftCollision = arr
                dataLeftCollisionN += 1
            else:
                dataLeftCollision = np.vstack((dataLeftCollision, arr))
                dataLeftCollisionN += 1

    if dbg:
        print('Normal= ' + str(dataNormalN) + '; frontCollision= ' + str(dataFrontCollisionN) + \
            '; backCollision= ' + str(dataBackCollisionN) + '; leftCollision= ' + str(dataLeftCollisionN) + \
            '; rightCollision= ' + str(dataRightCollisionN))
        print('Related entries = ' + str(dataNormalN + dataFrontCollisionN + dataBackCollisionN + \
            dataLeftCollisionN + dataRightCollisionN))

    if dbg:
        print('Loading lp3.data')
    with open("lp3.data") as f:
        content = f.readlines()
    content.reverse()
    if dbg:
        print('# of lines = ' + str(len(content)) + '; spected ' + str(len(content)/18.) + ' entries')
    tmp = '\n'
    while len(content) > 0:
        tmp = content.pop()
        while tmp == '\n':
            if len(content) > 0:
                tmp = content.pop()
            else:
                break;
        if tmp == 'ok\n':
            arr = readMeasure()
            if dataNormalN == 0:
                dataNormal = arr
                dataNormalN += 1
            else:
                dataNormal = np.vstack((dataNormal, arr))
                dataNormalN += 1

        elif tmp == 'slightly_moved\n':
            arr = readMeasure()
            if dataSliMovedN == 0:
                dataSliMoved = arr
                dataSliMovedN += 1
            else:
                dataSliMoved = np.vstack((dataSliMoved, arr))
                dataSliMovedN += 1

        elif tmp == 'moved\n':
            arr = readMeasure()
            if dataMovedN == 0:
                dataMoved = arr
                dataMovedN += 1
            else:
                dataMoved = np.vstack((dataMoved, arr))
                dataMovedN += 1

        elif tmp == 'lost\n':
            arr = readMeasure()
            if dataLostN == 0:
                dataLost = arr
                dataLostN += 1
            else:
                dataLost = np.vstack((dataLost, arr))
                dataLostN += 1

    if dbg:
        print('Normal= ' + str(dataNormalN) + '; slightly moved= ' + str(dataSliMovedN) + \
            '; moved= ' + str(dataMovedN) + '; lost= ' + str(dataLostN) )
        print('Related entries = ' + str(dataNormalN + dataFrontCollisionN + dataBackCollisionN + \
            dataLeftCollisionN + dataRightCollisionN))

    if dbg:
        print('Loading lp4.data')
    with open("lp4.data") as f:
        content = f.readlines()
    content.reverse()
    if dbg:
        print('# of lines = ' + str(len(content)) + '; spected ' + str(len(content)/18.) + ' entries')
    tmp = '\n'
    #ipdb.set_trace()
    while len(content) > 0:
        tmp = content.pop()
        while tmp == '\n':
            if len(content) > 0:
                tmp = content.pop()
            else:
                break;
        if tmp == 'normal\n':
            arr = readMeasure()
            if dataNormalN == 0:
                dataNormal = arr
                dataNormalN += 1
            else:
                dataNormal = np.vstack((dataNormal, arr))
                dataNormalN += 1

        elif tmp == 'collision\n':
            arr = readMeasure()
            if dataCollisionN == 0:
                dataCollision = arr
                dataCollisionN += 1
            else:
                dataCollision = np.vstack((dataCollision, arr))
                dataCollisionN += 1

        elif tmp == 'obstruction\n':
            arr = readMeasure()
            if dataObstructionN == 0:
                dataObstruction = arr
                dataObstructionN += 1
            else:
                dataObstruction = np.vstack((dataObstruction, arr))
                dataObstructionN += 1

    if dbg:
        print('Normal= ' + str(dataNormalN) + '; collision= ' + str(dataCollisionN) + \
            '; obstruction= ' + str(dataObstructionN))
        print('Related entries = ' + str(dataNormalN + dataCollisionN + dataObstructionN))

    if dbg:
        print('Loading lp5.data')
    with open("lp5.data") as f:
        content = f.readlines()
    content.reverse()
    if dbg:
        print('# of lines = ' + str(len(content)) + '; spected ' + str(len(content)/18.) + ' entries')
    tmp = '\n'
    #ipdb.set_trace()
    while len(content) > 0:
        #print(len(content))
        tmp = content.pop()
        while tmp == '\n':
            if len(content) > 0:
                tmp = content.pop()
            else:
                break;
        if tmp == 'normal\n':
            arr = readMeasure()
            if dataNormalN == 0:
                dataNormal = arr
                dataNormalN += 1
            else:
                dataNormal = np.vstack((dataNormal, arr))
                dataNormalN += 1

        elif tmp == 'bottom_collision\n':
            arr = readMeasure()
            if dataBottCollisionN == 0:
                dataBottCollision = arr
                dataBottCollisionN += 1
            else:
                dataBottCollision = np.vstack((dataBottCollision, arr))
                dataBottCollisionN += 1

        elif tmp == 'bottom_obstruction\n':
            arr = readMeasure()
            if dataBottObstructionN == 0:
                dataBottObstruction = arr
                dataBottObstructionN += 1
            else:
                dataBottObstruction = np.vstack((dataBottObstruction, arr))
                dataBottObstructionN += 1

        elif tmp == 'collision_in_part\n':
            arr = readMeasure()
            if dataPartCollisionN == 0:
                dataPartCollision = arr
                dataPartCollisionN += 1
            else:
                dataPartCollision = np.vstack((dataPartCollision, arr))
                dataPartCollisionN += 1

        elif tmp == 'collision_in_tool\n':
            arr = readMeasure()
            if dataToolCollisionN == 0:
                dataToolCollision = arr
                dataToolCollisionN += 1
            else:
                dataToolCollision = np.vstack((dataToolCollision, arr))
                dataToolCollisionN += 1

    if dbg:
        print('Normal= ' + str(dataNormalN) + '; bottom collision= ' + str(dataBottCollisionN) + \
            '; bottom obstruction= ' + str(dataBottObstructionN) + '; collision in part' + \
            str(dataPartCollisionN) + '; collision in tool' + str(dataToolCollisionN))
        print('Related entries = ' + str(dataNormalN + dataBottCollisionN + dataBottObstructionN + \
            dataPartCollisionN + dataToolCollisionN))

        print('Total entries =' + str(dataNormalN + dataCollisionN + dataFrontCollisionN + dataBackCollisionN + \
            dataLeftCollisionN + dataRightCollisionN + dataObstructionN + dataMovedN + dataSliMovedN + \
            dataLostN + dataBottCollisionN + dataBottObstructionN + dataPartCollisionN + dataToolCollisionN) )

    X = np.vstack((dataNormal, dataCollision, dataFrontCollision, dataBackCollision, \
        dataLeftCollision, dataRightCollision, dataObstruction, dataMoved, dataSliMoved, \
        dataLost, dataBottCollision, dataBottObstruction, dataPartCollision, dataToolCollision))

    y = np.concatenate((np.ones(dataNormalN)*1, np.ones(dataCollisionN)*2, np.ones(dataFrontCollisionN)*3, \
        np.ones(dataBackCollisionN)*4, np.ones(dataLeftCollisionN)*5, np.ones(dataRightCollisionN)*6, \
        np.ones(dataObstructionN)*7, np.ones(dataMovedN)*8, np.ones(dataSliMovedN)*9, \
        np.ones(dataLostN)*10, np.ones(dataBottCollisionN)*11, np.ones(dataBottObstructionN)*12, \
        np.ones(dataPartCollisionN)*13, np.ones(dataToolCollisionN)*14 ) )

    h0 = np.where(y == 1)
    h0 = h0[0]

    #ipdb.set_trace()
    svc.fit(X, y)
    p1 = svc.predict(X)
    print('SVM Total Hit score = ' + str(sum(p1 == y)*100./len(y)) )
    h1 = np.where(p1 == 1)
    h1 = h1[0]
    print('SVM Normal Hit score = ' + str(len(np.intersect1d(h1, h0))*100./len(h0)) )
    sgd.fit(X, y)
    p2 = sgd.predict(X)
    print('SGD Total Hit score = ' + str(sum(p2 == y)*100./len(y)) )
    h2 = np.where(p2 == 1)
    h2 = h2[0]
    print('SGD Normal Hit score = ' + str(len(np.intersect1d(h2, h0))*100./len(h0)) )
    clf.fit(X, y)
    p3 = clf.predict(X)
    print('DecisionTree Total Hit score = ' + str(sum(p3 == y)*100./len(y)) )
    h3 = np.where(p3 == 1)
    h3 = h3[0]
    print('DecisionTree Normal Hit score = ' + str(len(np.intersect1d(h3,h0))*100./len(h0)) )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Input Error - usage: ' + sys.argv[0] + ' measureFilename.txt')
    else:
    	dict = {1: 'Normal', 2 : 'Collision', 3 : 'Front Collision', 4 : 'Back Collision', 5 : 'Left Collision', \
    	6 : 'Right Collision', 7 : 'Obstruction', 8 : 'Moved', 9 : 'Slightly Moved', 10 : 'Lost', \
    	11 : 'Bottom Collision', 12 : 'Bottom Obstruction', 13 : 'Part Collision', 14 : 'Tool Collision'}
        #ipdb.set_trace()
        init(dbg=False)
        with open(sys.argv[1]) as f:
        	content = f.readlines()
        arr = readMeasure()
        #print(arr)
        p = clf.predict(arr)
        print('State = ' + str(p[0]) + ' : ' + dict[p[0]])