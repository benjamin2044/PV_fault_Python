
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

dataset = pd.read_csv('Solar_categorical.csv')
X = dataset.iloc[:3000, 0:7].values
y = dataset.iloc[:3000, 7].values
print(y)


###########################VISUALIZATION#################################################################
###########################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
 
#The ratio of train and validation set 
allLabels = np.concatenate((y_train, y_test))
cat1 = np.repeat('training',len(y_train))
cat2 = np.repeat('validation',len(y_test))
cat = np.concatenate((cat1,cat2))
hist_df = pd.DataFrame(({'labels':allLabels, 'datatype':cat}))
p = sb.countplot(data=hist_df,x='labels',hue='datatype',saturation=1,palette=['c', 'm'])
leg = p.get_legend()
leg.set_title("")
labs = leg.texts
labs[0].set_text("Training")
labs[1].set_text("Validation")
plt.xlabel('labels', fontsize=20)
plt.ylabel('count', fontsize=20)
#####################################################################################


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
encoder= LabelEncoder()
X[:,6] = encoder.fit_transform(X[:, 6])
#y = encoder.fit_transform(y)
#y_original = encoder.inverse_transform(y_encoded)
#y = to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
#y_original = encoder.inverse_transform(y_encoded)
#########################################################################################
from sklearn.svm import SVC
svc_clf = SVC(kernel="linear", probability=True)
svc_clf.fit(X_train, y_train)
            
y_pred = svc_clf.predict(X_test)
y_pred_label = encoder.inverse_transform(y_pred)
print(y_pred_label)

y_test_label = encoder.inverse_transform(y_test)
print(y_test_label)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_label, y_pred_label)

cm_fig = pd.DataFrame(cm, columns=np.unique(y_test_label), index=np.unique(y_test_label))
sb.set(font_scale=1.4)
sb.heatmap(cm_fig, cmap="RdBu_r", annot=True, annot_kws={"size":20}) 
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')

#new_pred1 = svc_clf.predict(np.array([[2.1, 3.1, 1.1, 43, 99]]))
#pred1_label = encoder.inverse_transform(new_pred1)
#print(pred1_label)
#
#new_pred2 = svc_clf.predict(np.array([[4.1, 3.5, 4.6, 45, 100]]))
#pred2_label = encoder.inverse_transform(new_pred2)
#print(pred2_label)
#
#new_pred3 = svc_clf.predict(np.array([[0, 4, 0.4, 15, 64]]))
#pred3_label = encoder.inverse_transform(new_pred3)
#print(pred3_label)
###################################################################################
######################################Evaluation###################################
from sklearn.metrics import f1_score

def evaluate(labelsTrue, predictions):
    if len(predictions)>0:
        f1 = f1_score(labelsTrue,predictions, average="weighted")
        print("F1 score: ",f1)
   
pred_svc = svc_clf.predict(X_test)
evaluate(y_test, pred_svc)
###############################################################################################
###############################################################################################
################################Compare several classifiers####################################
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

pred_knn = knn_clf.predict(X_test)
evaluate(y_test, pred_knn)
#############################################################
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=50,n_jobs=-1)
forest_clf.fit(X_train, y_train)

pred_forest = forest_clf.predict(X_test)
evaluate(y_test, pred_forest)
#########################################################
from sklearn.ensemble import ExtraTreesClassifier
trees_clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
trees_clf.fit(X_train, y_train)

pred_trees = trees_clf.predict(X_test)
evaluate(y_test, pred_trees)
##########################################################
from sklearn.ensemble import  AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train)

pred_ada = ada_clf.predict(X_test)
#evaluate(y_test, pred_ada)
f1_score(y_test, pred_ada, average='weighted', labels=np.unique(pred_ada))
#########################################################
from sklearn.naive_bayes import GaussianNB
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, y_train)

pred_bayes = bayes_clf.predict(X_test)
evaluate(y_test, pred_bayes)
######################################################################################################
###########################################Correlations between predicted classes#####################
predictions = pd.DataFrame( {'Rand_For': pred_forest,'KNear_Neigh': pred_knn,'Sup_Vec_Mac': pred_svc,
                             'ExtraTrees': pred_trees, 'AdaBoost': pred_ada,'NaiveBayes': pred_bayes})

sb.heatmap(predictions.corr(), linewidths=0.5, vmax=1.0, square=True, cmap='jet', linecolor='white', annot=True)
########################################################################################################


