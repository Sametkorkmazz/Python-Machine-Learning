
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold,cross_validate

from sklearn.neural_network import MLPClassifier


scoring = ['precision_macro', 'recall_macro']
os.chdir("C:/Users/debim/Desktop/")
wines = pd.read_csv("winequality-white.csv",sep=";")


X = wines.iloc[:,:-1]
y = wines.iloc[:,-1]


X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,
                                                   random_state=0,stratify=y)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
test_sonuc_tree = clf.predict(X_test)

cm_tree = confusion_matrix(y_test, test_sonuc_tree,labels=[0,1,2,3,4,5,6,7,8,9,10])
plt.matshow(cm_tree)
plt.title('Confusion matrix With Decision Tree')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
kfolds = KFold(n_splits = 10)
print(f"Decision Tree Doğruluk Oranı: {accuracy_score(test_sonuc_tree,y_test)}")
tree_10_cross_acc = cross_val_score(clf,X_train,y_train,cv=kfolds)
tree_recall_prec = cross_validate(clf, X_train,y_train,cv=kfolds,scoring=["precision_macro","recall_macro"])
print("Decision Tree 10 katlı çarpraz doğrulama:\nAccuracy: ")
for acc in tree_10_cross_acc:
    print("%.7f "%acc,end="")
print("\nRecall:")
for rec in tree_recall_prec["test_recall_macro"]:
   print("%.7f "%rec,end="")
print("\nPrecision:")
for pre in tree_recall_prec["test_precision_macro"]:
   print("%.7f "%pre,end="")
print("\n\nConfusion Matrix:")
print(cm_tree)

mlp = MLPClassifier(random_state=0, max_iter=100)
y_pred = mlp.fit(X_train,y_train).predict(X_test)
cm_mlp = confusion_matrix(y_test, y_pred,labels=[0,1,2,3,4,5,6,7,8,9,10])
print(f"\nMLP Classifier YSA Doğruluk Oranı: {accuracy_score(y_pred,y_test)}")
MLP_10_cross_acc = cross_val_score(mlp,X_train,y_train,cv=kfolds)
MLP_recall_prec = cross_validate(mlp, X_train,y_train,cv=kfolds,scoring=["precision_macro","recall_macro"])
print("MLP Classifier 10 katlı çarpraz doğrulama:\nAccuracy: ")
for acc in MLP_10_cross_acc:
    print("%.7f "%acc,end="")
print("\nRecall:")
for rec in MLP_recall_prec["test_recall_macro"]:
    print("%.7f "%rec,end="")
print("\nPrecision:")
for pre in MLP_recall_prec["test_precision_macro"]:
    print("%.7f "%pre,end="")
print("\nConfusion Matrix:")
print(cm_mlp)

plt.matshow(cm_mlp)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix with MLP Classifier')
plt.show()


print(clf.predict([[4.3,0.25,0.1,25,0.054,20,150,0.972,3.2,0.57,7]]))
print(mlp.predict([[4.3,0.25,0.1,25,0.054,20,150,0.972,3.2,0.57,7]]))

print(clf.predict([[3.2,0.45,0.3,50,0.059,15,70,0.989,3.5,0.43,9]]))
print(mlp.predict([[3.2,0.45,0.3,50,0.059,15,70,0.989,3.5,0.43,9]]))

input()




