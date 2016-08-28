import time
import numpy as np
from sklearn.svm import SVR
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ipdb

### global variables ###
# change fileName to select other data
#fileName = 'data/ethylene_methane.txt'
fileName = 'data/ethylene_CO.txt'
frequencySample = 100
minInterval = 80    # seconds
up_perc = int(0.25*minInterval*frequencySample)
dn_perc = int(0.1*minInterval*frequencySample)
skipRows = 1001
toDebbug = True

def getChanges():
    '''
    get changes in concentration
    '''
    global up_perc
    global dn_perc
    global fileName
    global skipRows
    methane = np.loadtxt(fileName, skiprows=skipRows, usecols=(1,), dtype='float32')
    md = np.diff(methane)
    md_id = np.where(md > 0)
    md_up = np.array(md_id[0], dtype='int32')
    md_id = np.where(md < 0)
    md_dn = np.array(md_id[0], dtype='int32')
    m = np.r_[methane[md_up+up_perc], methane[md_dn-dn_perc]]
    mid = np.r_[md_up+up_perc, md_dn-dn_perc]
    md_up0 = md_up[np.where((methane[md_up] == 0))]
    ethylene = np.loadtxt(fileName, skiprows=skipRows, usecols=(2,), dtype='float32')
    ed = np.diff(ethylene)
    ed_id = np.where(ed > 0)
    ed_up = np.array(ed_id[0], dtype='int32')
    ed_id = np.where(ed < 0)
    ed_dn = np.array(ed_id[0], dtype='int32')
    e = np.r_[ethylene[ed_up+up_perc], ethylene[ed_dn-dn_perc]]
    eid = np.r_[ed_up+up_perc, ed_dn-dn_perc]
    ed_up0 = ed_up[np.where((ethylene[ed_up] == 0))]

    me = np.r_[ethylene[md_up+up_perc], ethylene[md_dn-dn_perc]]
    em = np.r_[methane[ed_up+up_perc], methane[ed_dn-dn_perc]]

    # find both gases equal zero
    idG0 = list()
    for i in md_up0:
        for j in ed_up0:
            if i == j:
                idG0.append(i)
    if len(idG0) == 0:
        print('Any Ar only concentration is found!')

    del methane
    del ethylene
    return (m, mid, me, e, eid, em, idG0)

def getSensorData(ids, idG0, nsensor=16):
    '''
    get Sensor changes in concentration
    '''
    # improve when find more than one Ar only concentration
    global fileName
    data = np.array([])
    G0 = list()
    for i in range(nsensor):
        sensor = np.loadtxt(fileName, skiprows=skipRows, usecols=(i+3,), dtype='float32')
        G0.append(sensor[i])
        if len(data) == 0:
            data = sensor[ids]
        else:
            data = np.row_stack((data, sensor[ids]))
    #if len(idG0) > 0:
    #    G0 = list()
    #    for i in idG0:
    #        G0.append(sensor[i])
    return (data, G0)

def cliffordTuma(ratio, b, beta):
    '''
        Sensor conductivity (GS) as a function of
        the gas concentration (c) and the sensor conductivity in air (G0).
    '''
    return (np.log(ratio)/np.log(beta)-1.)/b

def findCliffordTumaCoeff(data, p0=(-0.002, 0.1)):
    '''
    '''
    # improve when find more than one Ar only concentration
    global toDebbug
    global nsensor
    print('... Curve Fit')
    if len(data[0]) != len(data[2]):
        print('Vector length are different!')
    if len(data[3]) == 1:
        print('Only one Ar only concentration!')
    ipdb.set_trace()
    popt_list = list()
    pcov_list = list()
    for i in range(nsensor):
        popt, pcov = curve_fit(cliffordTuma, data[2][i]/data[3][i], data[0], p0, maxfev=10000)
        popt_list.append(popt)
        pcov_list.append(pcov)
    #if toDebbug:
    #    print(popt)
    #    for i in range(len(data[0])):
    #        print('...: S={0:.2f}; G0={1:.2f}; G1={2:.2f}'.format(data[0][i], data[3][0], data[2][i]))
    # returning (b, beta) 
    return (popt_list, pcov_list)

def cliffordTumaMix(ratio, c2, gas1, gas2):
    '''
        return gas1 concentration (c1) as function of gas2 concentration (c2)
    '''
    lg = np.log(ratio)/np.log(gas1[1]) # beta1
    aux = ( ( 1 + gas2[0]*c2 )**gas2[1] )**(1./gas1[1]) # b2, beta2, beta1
    return (lg - aux)/gas1[0] # b1


### MAIN ###
nsensor = 16

print(' --- Gas Sensor Analysis ---')
print('search for gas concentration changes ...')
start = time.time()
m, mid, me, e, eid, em, idG0 = getChanges()
print('Elapsed time = {0:.2f} sec'.format(time.time() - start))
print(':::{0} changes in gas1 conc. and {1} in ethylene conc.'.format(len(m), len(e)))
print('get {0} sensor data related with concentration changes ...'.format(nsensor))
start = time.time()
ms, mG0 = getSensorData(mid, idG0, nsensor)
es, eG0 = getSensorData(eid, idG0, nsensor)
print('Elapsed time = {0:.2f} sec'.format(time.time() - start))
#if mG0 == eG0:
#    print('G0 equal for both gases!')
#else:
#    print('G0 NOT equal for both gases!')
print('Find Clifford Tuma Coeff ...')
methane = (m, mid, ms, mG0)
ethylene = (e, eid, es, eG0)
mcoeffs = findCliffordTumaCoeff(methane, (-0.004, 0.1))
ecoeffs = findCliffordTumaCoeff(ethylene, (-0.1, 0.1))

#for i in range(nsensor):
#    if ecoeffs[i][0][0] > -0.5:
#        print('Ethylene b coefficient too high in sensor {0}, Assuming -0.1'.format(i))
#        ecoeffs = (np.array([-0.1, ecoeffs[i][0][1]]), ecoeffs[i][1])

#print('Methane coeff : b = {0:}, beta = {1:}'.format(mcoeffs[0][0], mcoeffs[0][1]))
#print('Ethylene coeff : b = {0:}, beta = {1:}'.format(ecoeffs[0][0], ecoeffs[0][1]))

print('fit individual gas concentration ...')
clf1 = SVR(C=1.0, epsilon=0.2)
clf2 = SVR(C=1.0, epsilon=0.2)
start = time.time()
clf1.fit(np.transpose(ms), m)
clf2.fit(np.transpose(es), e)

mp = clf1.predict(np.transpose(ms))
ep = clf2.predict(np.transpose(es))
print('Elapsed time = {0} sec'.format(time.time() - start))

merr1 = mp - m
eerr1 = ep - e

merr = np.array([])
eerr = np.array([])
for i in range(nsensor):
    gas1 = (ecoeffs[0][i][0], ecoeffs[0][i][1])
    gas2 = (mcoeffs[0][i][0], mcoeffs[0][i][1])
    c2 = cliffordTumaMix(ms[i]/mG0[i], mp, gas1, gas2)
    c1 = cliffordTumaMix(es[i]/eG0[i], ep, gas2, gas1)
    print('mse : c1 = {0}; c2={1}'.format(np.sqrt(sum(c1**2)), np.sqrt(sum(c2**2))))
    if len(merr) == 0:
        merr = c2
        eerr = c1
    else:
        merr = np.row_stack((merr, c2))
        eerr = np.row_stack((eerr, c1))
f, ax = plt.subplots()
ax.plot(np.transpose(merr))
f, ax = plt.subplots()
ax.plot(np.transpose(eerr))
plt.show()
