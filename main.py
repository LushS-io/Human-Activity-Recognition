# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# We are using Smartphone-Based Recognition of Human Activities 
# and Postural Transitions Data Set Version 2.1

# WE DID NOT GENERATE THIS DATA!

# License:
# ================================================================================
# Use of this dataset in publications must be acknowledged by 
# referencing the following publications

# - Jorge-L. Reyes-Ortiz, Luca Oneto, Albert Samà, Xavier Parra, 
# Davide Anguita. Transition-Aware Human Activity Recognition Using 
# Smartphones. Neurocomputing. Springer 2015.

# ================================================================================

# This dataset is distributed AS-IS and no responsibility 
# implied or explicit can be addressed to the authors or their 
# institutions for its use or misuse. Any commercial use is prohibited.

# For more information please consult the README.txt included in this project
# and consult www.smartlab.ws for any and all questions comments concerns regarding
# the data.

# ================================================================================

# The following analysis is the work of Christopher Roberts and Troy Kirin 
# for the final project of the CptS_315: Data Mining class at Washington State 
# University (WSU)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Programmers: Christopher Roberts & Troy Kirin
# Assignment: Final Project of Data Mining (CptS_315)
# Date began: April 6th, 2019 -- (4/6/2019)
# Date finished: TBD

# %% imports
import modin.pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# %% crate label dataframe, NAMES and NUMBER
labelstxt = pd.read_csv(filepath_or_buffer="./HAPT Data Set/activity_labels.txt", names=['label'])

labels = pd.DataFrame(columns=['labels', 'numbers'])
labels['labels'] = labelstxt['label'].apply(lambda x: [str(x).strip(' 0123456789')])
labels['numbers'] = labelstxt['label'].apply(lambda x: [str(x).strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ_ ')])
print(labels)

# %% create data frames
df_features_info = pd.read_csv(filepath_or_buffer='./HAPT Data Set/features.txt', names=['features'])
listfeatures = df_features_info['features']

df_train = pd.read_csv(filepath_or_buffer='./HAPT Data Set/Train/X_train.txt', sep=' ', names=listfeatures)
df_train_labels = pd.read_csv(filepath_or_buffer='./HAPT Data Set/Train/y_train.txt', names=['label'])
df_test = pd.read_csv(filepath_or_buffer='./HAPT Data Set/Test/X_test.txt', sep=' ', names=listfeatures)
df_test_labels = pd.read_csv(filepath_or_buffer='./HAPT Data Set/Test/y_test.txt', names=['label'])

print("df_train shape: " + str(df_train.shape))
print("df_test shape: " + str(df_test.shape))

# %% Preprocess the X data by scaling
sc = StandardScaler()
sc.fit(df_train)
x_train_std = sc.transform(df_train)
x_test_std = sc.transform(df_test)

# %% train the learners and predict
# PerceptronList = []
# SVCList = []
# RandomForestList = []
# EnsambleList = []
# for x in range(50, 1001, 50):
#     print(x)
#     ppn = Perceptron(max_iter=x, eta0=0.1, random_state=0)
#     ppn.fit(x_train_std, df_train_labels['label'])
#     y_pred = ppn.predict(x_test_std)
#     PerceptronList.append(accuracy_score(df_test_labels['label'], y_pred))

#     clf = LinearSVC(random_state=0, tol=1e-5, max_iter=x)
#     clf.fit(x_train_std, df_train_labels['label'])
#     y_pred = clf.predict(x_test_std)
#     SVCList.append(accuracy_score(df_test_labels['label'], y_pred))

#     rand_f = RandomForestClassifier()
#     rand_f.fit(x_train_std, df_train_labels['label'])
#     y_pred = rand_f.predict(x_test_std)
#     RandomForestList.append(accuracy_score(df_test_labels['label'], y_pred))

# print("Perceptron accuracy")
# print(PerceptronList)
# print("SVM accuracy")
# print(SVCList)
# print("Random Forest accuracy")
# print(RandomForestList)
scoresList = []
for x in range(1, 21):
    print(x)
    ppn = Perceptron(max_iter=x, eta0=0.1, random_state=0, tol=1e-3)
    # ppn.fit(x_train_std, df_train_labels['label'])
    # y_pred = ppn.predict(x_test_std)
    # PerceptronList.append(accuracy_score(df_test_labels['label'], y_pred))

    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=x)
    # clf.fit(x_train_std, df_train_labels['label'])
    # y_pred = clf.predict(x_test_std)
    # SVCList.append(accuracy_score(df_test_labels['label'], y_pred))

    rand_f = RandomForestClassifier(n_estimators=x)
    # rand_f.fit(x_train_std, df_train_labels['label'])
    # y_pred = rand_f.predict(x_test_std)
    # RandomForestList.append(accuracy_score(df_test_labels['label'], y_pred))

    eclf = VotingClassifier(estimators=[('per', ppn), ('svm', clf), ('rf', rand_f)], voting='hard')

    for classifier, label in zip([ppn, clf, rand_f, eclf], ['Perceptron', 'SVM', 'Random Forest', 'Ensemble']):
        classifier.fit(x_train_std, df_train_labels['label'])
        y_pred = classifier.predict(x_test_std)
        score = (x, label, accuracy_score(df_test_labels['label'], y_pred))
        scoresList.append(score)
        # scores = cross_val_score(classifier, x_train_std, df_train_labels['label'], cv=1, scoring='accuracy')
        # scoresList.append((label, scores, x))
        # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

for item in scoresList:
    print(item)