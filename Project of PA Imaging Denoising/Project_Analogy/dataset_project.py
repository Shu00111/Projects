import numpy as np
import scipy
from scipy import io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.fftpack import fft

X_data = [0 for i in range(8)]
X_data[0] = scipy.io.loadmat('Signal_withnoise/brain1_X.mat')['X']
X_data[1] = scipy.io.loadmat('Signal_withnoise/brain2_X.mat')['X']
X_data[2] = scipy.io.loadmat('Signal_withnoise/brain3_X.mat')['X']
X_data[3] = scipy.io.loadmat('Signal_withnoise/brain4_X.mat')['X']
X_data[4] = scipy.io.loadmat('Signal_withnoise/brain5_X.mat')['X']
X_data[5] = scipy.io.loadmat('Signal_withnoise/brain6_X.mat')['X']
X_data[6] = scipy.io.loadmat('Signal_withnoise/brain7_X.mat')['X']
X_data[7] = scipy.io.loadmat('Signal_withnoise/brain8_X.mat')['X']

Y_data = [0 for i in range(8)]
Y_data[0] = scipy.io.loadmat('Ground_truth/brain1_Y.mat')['Y']
Y_data[1] = scipy.io.loadmat('Ground_truth/brain2_Y.mat')['Y']
Y_data[2] = scipy.io.loadmat('Ground_truth/brain3_Y.mat')['Y']
Y_data[3] = scipy.io.loadmat('Ground_truth/brain4_Y.mat')['Y']
Y_data[4] = scipy.io.loadmat('Ground_truth/brain5_Y.mat')['Y']
Y_data[5] = scipy.io.loadmat('Ground_truth/brain6_Y.mat')['Y']
Y_data[6] = scipy.io.loadmat('Ground_truth/brain7_Y.mat')['Y']
Y_data[7] = scipy.io.loadmat('Ground_truth/brain8_Y.mat')['Y']

plt.pcolormesh(X_data[0])
plt.show()

plt.pcolormesh(Y_data[0])
plt.show()

X_data[0] = np.array(X_data[0])
fft1 = scipy.fftpack.fft(X_data[0])
fft1 = abs(fft1)
plt.plot(abs(fft1))
plt.show()

pca1 = PCA(n_components=1)
pca1.fit(fft1)
pca1_ = pca1.transform(fft1)
plt.plot(pca1_)
plt.show()

Y_data[0] = np.array(Y_data[0])
fft2 = scipy.fftpack.fft(Y_data[0])
fft2 = abs(fft2)
plt.plot(fft2)
plt.show()

pca2 = PCA(n_components=1)
pca2.fit(fft2)
pca2_ = pca2.transform(fft2)
plt.plot(pca2_)
plt.show()

noise1 = abs(fft1 - fft2)
plt.plot(abs(noise1))
plt.show()
pca3 = PCA(n_components=1)
pca3.fit(noise1)
pca3_ = pca2.transform(noise1)
plt.plot(pca3_)
plt.show()

