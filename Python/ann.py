import numpy as np
from keras.utils import np_utils
from Python.labels import get_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=15, activation='elu'))
    model.add(Dense(30, activation='elu'))
    model.add(Dense(6, activation='softmax')) # softmax because multiclass classification
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Import data
dataframe = np.genfromtxt('freq_components.csv', delimiter=',')
dataframe = StandardScaler().fit_transform(dataframe)
label_names = get_labels()
# Get rid of 'Asthma' and 'LRTI' (only 1 and 2 data points respectively in dataset)
index_to_delete = label_names.index('Asthma')
del(label_names[index_to_delete])
dataframe = np.delete(dataframe, index_to_delete, 0)
for i in range(2):
    index_to_delete = label_names.index('LRTI')
    del (label_names[index_to_delete])
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
