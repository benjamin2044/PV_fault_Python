
### 3 Faults selection ###
## Solar_data_Experiment

########### Part1 (DATA PRE-PROCESSING) ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

dataset = pd.read_csv('Solar_categorical.csv')
X = dataset.iloc[:3000, 0:7].values
y = dataset.iloc[:3000, 7].values
#print(y)

"""
###### Input data Data Visualization #####

plt.xlabel('Sample size', fontsize = 10)
plt.ylabel('Current (A)', fontsize = 10)
plt.title('Summer features', fontsize = 12)
plt.plot(X[:250-1, 0], label='Normal Sunny') # Summer normal sunny
plt.plot(X[251:501-1, 0], label='Normal Cloudy') # Summer normal cloudy
plt.plot(X[1002:1249-1, 1], label='Line-line Sunny') # Summer line-line cloudy
plt.plot(X[1251:1501-1, 0], label='Line-line Cloudy') # Summer line-line cloudy
plt.legend(fancybox=True, shadow=True)

plt.xlabel('Sample size', fontsize = 10)
plt.ylabel('Current (A)', fontsize = 10)
plt.title('Winter features', fontsize = 12)
plt.plot(X[1502:1751-1, 0], label='Normal Sunny') # Winter normal sunny
plt.plot(X[1752:2001, 0], label='Normal Cloudy') # Winter normal cloudy
plt.plot(X[2502:2750, 1], label='Line-line Sunny') # Winter line-line sunny
plt.plot(X[2751:3001, 1], label='Line-line Cloudy') # Winter normal cloudy (temp 4 deg to -5,)
plt.legend(fancybox=True, shadow=True)

#Plotting in subplots
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle( "Summer and winter features")
ax1.plot(X[:250-1, 0], label='Normal Sunny')
ax1.plot(X[251:501-1, 0], label='Normal Cloudy') # Summer normal cloudy
ax1.plot(X[1002:1249-1, 1], label='Line-line Sunny') # Summer line-line cloudy
ax1.plot(X[1251:1501-1, 0], label='Line-line Cloudy') # Summer line-line cloudy
ax1.legend()

ax2.plot(X[1502:1751-1, 0], label='Normal Sunny') # Winter normal sunny
ax2.plot(X[1752:2001, 0], label='Normal Cloudy') # Winter normal cloudy
ax2.plot(X[2502:2750, 1], label='Line-line Sunny') # Winter line-line sunny
ax2.plot(X[2751:3001, 1], label='Line-line Cloudy') # Winter normal cloudy (temp 4 deg to -5,)
ax2.legend()
"""

########## Label Encoding categorical data  ###########
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
encoder= LabelEncoder()
X[:,6] = encoder.fit_transform(X[:, 6])
y = encoder.fit_transform(y)
#y_original = encoder.inverse_transform(y_encoded)
y = to_categorical(y)

##########################################################################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#####################################################################################

# Feature Scaling (To scale all variables to similar scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


############################################################################
########### Part2 (DEEP NEURAL NETWORK)/Multiple Layer Perceptron(MLP)#########################
from keras.models import Sequential  ##module to initialize ANN
from keras.layers import Dense #module required to build layers

DNN_model = Sequential() 
DNN_model.add(Dense(input_dim = 7, units = 8, kernel_initializer = 'uniform', activation = 'relu'))
DNN_model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
DNN_model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
DNN_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(DNN_model.summary())

history = DNN_model.fit(X_train, y_train, batch_size = 5,  epochs = 200,
                        validation_data=(X_test, y_test), shuffle=True)
print(history.history.keys())

######Visualizing model: train & test accuracy and loss ############
"""
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

plt.plot(history.history['loss']) # summarize history for loss
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
"""
################### Part3 (Making predictions and evaluating the model)#################
y_pred = DNN_model.predict(X_test) # Predicting the Test set results
y_pred = (y_pred > 0.95)
y_pred_label = encoder.inverse_transform(np.argmax(y_pred, axis=1))
print(y_pred_label)

y_test_label = encoder.inverse_transform(np.argmax(y_test, axis=1))
print(y_test_label)

#Making confusion matrix that checks accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_label, y_pred_label)

#### Visualizing Confusion Matrix  ########
cm_fig = pd.DataFrame(cm, columns=np.unique(y_test_label), index=np.unique(y_test_label))
sb.set(font_scale=1.3)
sb.heatmap(cm_fig, cmap="RdBu_r", annot=True, annot_kws={"size":15}, fmt='.2f') 
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
#####################################################################
######### Predicting with new data ##########
new_pred1 = DNN_model.predict(sc.transform(np.array([[5, 6.2, 92, 93, 85, 16, 1]])))
new_pred1_original = encoder.inverse_transform([np.argmax(new_pred1)])
print(new_pred1_original)

new_pred2 = DNN_model.predict(sc.transform(np.array([[0, 3.5, 0, 95, 96, 12, 1]])))
new_pred2_original = encoder.inverse_transform([np.argmax(new_pred2)])
print(new_pred2_original)

new_pred3 = DNN_model.predict(sc.transform(np.array([[3.7, 0.2, 85, 65, 78, 7, 0]])))
new_pred3_original = encoder.inverse_transform([np.argmax(new_pred3)])
print(new_pred3_original)
#################################################################################

###Saving the model
DNN_model.save('fault_model.model')

### Opening the saved model
from keras.models import load_model
new_model = load_model('fault_model.model')
new_model.summary()

###### Predictiong with new data
new_mod_test = new_model.predict(sc.transform(np.array([[2.1, 3.1, 90, 83, 88, 15, 1]])))
new_mod_test_original = encoder.inverse_transform([np.argmax(new_mod_test)])
print(new_mod_test_original)
#####################################################

#Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# to check bias-variance tradeoff

def build_classifier():
    model = Sequential()
    model.add(Dense(input_dim = 7, units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 200)
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5) #CV = K-fold cross val. splits
mean = accuracies.mean()
variance = accuracies.std()
print("Average accuracy:", mean, "Variance:", variance)

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
from keras.layers import Dropout

model = Sequential()
model.add(Dense(input_dim = 7, units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 5, epochs = 200)

y_pred_drop = model.predict(X_test)
y_pred_drop_encoded = np.argmax(y_pred_drop, axis=1)
y_pred_drop_original = encoder.inverse_transform(y_pred_drop_encoded)
print(y_pred_drop_original)

# Tuning the ANN
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def build_network(optimizer):
    network = Sequential()
    network.add(Dense(input_dim = 5, units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    network.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    network.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
    network.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return network

network = KerasClassifier(build_fn = build_network, class_weight='balanced')
parameters = {'batch_size': [1, 2, 3], 'epochs': [100, 200, 300],
              'optimizer': ['sgd', 'rmsprop', 'adam', 'adagrad']}
grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 5)
grid_result = grid_search.fit(X_train, np.argmax(y_train, axis=1))
best_parameters = grid_result.best_params_
best_accuracy = grid_search.best_score_
################## END ######################


