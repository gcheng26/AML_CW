import numpy as np
from keras.utils import np_utils
from Python.labels import get_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix

num_of_features = 20

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=num_of_features, activation='elu'))
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
# Preprocess dataframe for 15 input nodes (reduce from 4096 to 15 features)
pca = PCA(n_components=num_of_features)
principalComponents = pca.fit_transform(dataframe)
features_and_labels = np.append(principalComponents, one_hot_labels, axis=1)
np.random.shuffle(features_and_labels)
X_train, X_test, y_train, y_test = train_test_split(features_and_labels[:, 0:num_of_features], features_and_labels[:, num_of_features:], test_size=0.25)


# Implement callback for early stopping
model = baseline_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model_'+str(num_of_features)+'_features.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# Train model
history = model.fit(X_train, y_train, validation_split=0.25, epochs=500, verbose=0, callbacks=[es, mc])

# evaluate the saved model
#saved_model = load_model('best_model_'+str(num_of_features)+'_features.h5')
saved_model = load_model('94percent_20_features.h5')
_, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# confusion matrix
y_pred = saved_model.predict(X_test)
y_pred_cat = np.zeros((len(y_pred),), dtype=int)
y_test_cat = np.zeros((len(y_pred),), dtype=int)
for i in range(len(y_pred)):
    max_index = np.argmax(y_pred[i])
    y_pred_cat[i] = max_index
    max_index = np.argmax(y_test[i])
    y_test_cat[i] = max_index
cm = confusion_matrix(y_test_cat, y_pred_cat)
print(cm)
