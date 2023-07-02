import scipy
from scipy import io
from matplotlib import pyplot as plt

Y_data = [0 for i in range(8)]
Y_data[0] = scipy.io.loadmat('Outcome/brain1_Y.mat')['Y']
Y_data[1] = scipy.io.loadmat('Outcome/brain2_Y.mat')['Y']
Y_data[2] = scipy.io.loadmat('Outcome/brain3_Y.mat')['Y']
Y_data[3] = scipy.io.loadmat('Outcome/brain4_Y.mat')['Y']
Y_data[4] = scipy.io.loadmat('Outcome/brain5_Y.mat')['Y']
Y_data[5] = scipy.io.loadmat('Outcome/brain6_Y.mat')['Y']
Y_data[6] = scipy.io.loadmat('Outcome/brain7_Y.mat')['Y']
Y_data[7] = scipy.io.loadmat('Outcome/brain8_Y.mat')['Y']

plt.pcolormesh(Y_data[0])
plt.show()
