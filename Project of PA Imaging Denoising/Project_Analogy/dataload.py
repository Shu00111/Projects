import numpy as np
import scipy
from scipy import io
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dataload():
    def __init__(self):
        self.X_data = None
        self.Y_data = None
    def Load(self):
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

        self.X_data = X_data
        self.Y_data = Y_data
        self.len_X = len(X_data)

        return self.X_data, self.Y_data, self.len_X










