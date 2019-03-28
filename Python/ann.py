import numpy as np
from keras.utils import np_utils
from Python.labels import get_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import data
dataframe = np.genfromtxt('freq_components.csv', delimiter=',')
dataframe = StandardScaler().fit_transform(dataframe)
label_names = get_labels()
# Get rid of 'Asthma' category because only one data point is available
index_to_delete = label_names.index('Asthma')
del(label_names[index_to_delete])
dataframe = np.delete(dataframe, index_to_delete, 0)
# Encode output variable
encoder = LabelEncoder()
encoder.fit(label_names)
encoded_labels = encoder.transform(label_names)
one_hot_labels = np_utils.to_categorical(encoded_labels)
# Preprocess dataframe for 15 input nodes
pca = PCA(n_components=15)
principalComponents = pca.fit_transform(dataframe)

#TODO: Build ANN - 15,20,20,7
