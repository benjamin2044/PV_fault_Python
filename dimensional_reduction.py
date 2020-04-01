
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


dataset = pd.read_csv('Solar_categorical.csv')
X = dataset.iloc[:3000, 0:7].values
y = dataset.iloc[:3000, 7].values

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
encoder= LabelEncoder()
X[:,6] = encoder.fit_transform(X[:, 6])

###################################################################################
#######  PCA (Unsupervised)############################################
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p_components = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

principalDF = pd.DataFrame(data = p_components, columns = ['principal component 1', 'principal component 2'])
principalDF = pd.concat([principalDF, dataset[['State']]], axis=1)

type_faults = ['Normal', 'Open', 'Line-line']
colors = ['r', 'g', 'b']

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('2 Component PCA', fontsize = 12)
for Fault, color in zip(type_faults, colors):
    indicesToKeep = principalDF['State'] == Fault
    ax.scatter(principalDF.loc[indicesToKeep, 'principal component 1'], 
               principalDF.loc[indicesToKeep, 'principal component 2'] , 
               c = color, s = 12)
ax.legend(type_faults)
###################################################################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.svm import SVC
svc_clf = SVC(kernel="linear", probability=True)
svc_clf.fit(X_train, y_train)
            
y_pred = svc_clf.predict(X_test)
y_pred_label = encoder.inverse_transform(y_pred)
print(y_pred_label)

y_test_label = encoder.inverse_transform(y_test)
print(y_test_label)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_label, y_pred_label)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc_clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Support Vector Machine (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc_clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Support Vector Machine (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

##############################################################################################
######################## LDA (Supervised) ###########################################

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
p_components = lda.fit_transform(X_scaled, y)
var_ratio = lda.explained_variance_ratio_

principalDF = pd.DataFrame(data = p_components, columns = ['principal component 1', 'principal component 2'])
principalDF = pd.concat([principalDF, dataset[['Fault']]], axis=1)

type_faults = ['Normal', 'Open', 'Short']
colors = ['r', 'g', 'b']

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component LDA', fontsize = 20)
for Fault, color in zip(type_faults, colors):
    indicesToKeep = principalDF['Fault'] == Fault
    ax.scatter(principalDF.loc[indicesToKeep, 'principal component 1'], 
               principalDF.loc[indicesToKeep, 'principal component 2'] , 
               c = color, s = 50)
ax.legend(type_faults)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explained_variance = lda.explained_variance_ratio_

from sklearn.svm import SVC
svc_clf = SVC(kernel="linear", probability=True)
svc_clf.fit(X_train, y_train)
            
y_pred = svc_clf.predict(X_test)
y_pred_label = encoder.inverse_transform(y_pred)
print(y_pred_label)

y_test_label = encoder.inverse_transform(y_test)
print(y_test_label)

## Predicting with new data #####
new_pred1 = svc_clf.predict(lda.transform(sc.transform(np.array([[2.1, 3.1, 1.2, 33, 92]]))))
new_pred1_original = encoder.inverse_transform(new_pred1)
print(new_pred1_original)

new_pred2 = svc_clf.predict(lda.transform(sc.transform(np.array([[4.1, 3.5, 4.6, 45, 100]]))))
new_pred2_original = encoder.inverse_transform(new_pred2)
print(new_pred2_original)

new_pred3 = svc_clf.predict(lda.transform(sc.transform(np.array([[0, 4, 0.4, 15, 64]]))))
new_pred3_original = encoder.inverse_transform(new_pred3)
print(new_pred3_original)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_label, y_pred_label)

# Kernel PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



































