"""
An Artificial Neural Network using only power spectrum to classify.
"""

import numpy as np
from Python.labels import get_labels
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

power_spectra = np.genfromtxt('freq_components.csv', delimiter=',')
labels = get_labels()

#TODO : PCA on 4096 features

power_spectra = StandardScaler().fit_transform(power_spectra)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(power_spectra)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)

targets = ['URTI', 'Healthy', 'Asthma', 'COPD', 'LRTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis']
colours = ['b', 'g', 'w', 'r', 'c', 'm', 'y', 'k']

for i in range(920):
    label = labels[i]
    if label != 'COPD':
        colour = colours[targets.index(label)]
        ax.scatter(principalComponents[i, 0]
                   , principalComponents[i, 1]
                   , principalComponents[i, 2]
                   , c=colour
                   , s=50)

ax.grid()
fig.show()