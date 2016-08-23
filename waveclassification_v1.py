import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
data = np.loadtxt("waveform.data", delimiter=',')
X = data[:500, :20]
y = data[:500, 21]
svc = svm.SVC(kernel='rbf', C=1, gamma='auto').fit(X, y)
Z = svc.predict(data[:, :20])
result = Z == data[:, 21]
print('percentage of accuracy = ' + str(sum(result)/5000.*100) + '%')

# create a mesh to plot in
h = .1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
XX = np.c_[xx.ravel(), yy.ravel()]
XX = np.concatenate([XX, np.zeros([len(XX), 18])], axis=1)
plt.subplot(1, 1, 1)
ZX = svc.predict(XX)
ZX = ZX.reshape(xx.shape)
plt.contourf(xx, yy, ZX, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('1th attribute')
plt.ylabel('2nd attribute')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rgb kernel')
plt.show()