# Importing tools
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
# Ignore warning 
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    



# Reading the documents texts in a list 'text_data'
text_data = []
path='Tobacco3482-OCR'
for file in glob.glob(os.path.join(path,'*')):
    for text_file in glob.glob(os.path.join(file,'*.txt')):
        with open(text_file, 'r') as myfile:
            text_data.append(myfile.read().replace('\n', '')) 
            
# Print the length of the text documents
len(text_data)


df = pd.read_csv('tobacco-lab_data_Tobacco3482.csv')
print("Shape :", df.shape)


for i, content in enumerate(text_data):  
    df.loc[i, 'img_path'] = content 
    
nans = df.isnull().sum() / len(df.index) # proportion de NaN dans chaque colonne
nans = nans.sort_values(ascending = False)
print("Attribut    Taux de NaN")
print(nans.to_string())

# Plot the statistics of labels
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
plot = sns.countplot(x='label', data=df, order = df['label'].value_counts().index, palette="Set3")

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[0]], df[df.columns[1]], test_size=0.30)
X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.50)

print("Number of train data :", len(X_train))
print("Number of dev data :", len(X_dev))
print("Number of test data :", len(X_test))


def bow_rep(X_train, X_dev, X_test):
    # Create document vectors
    count_vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(max_features=2000)
    vectorizer.fit(X_train)
    X_train_counts = vectorizer.transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    X_dev_counts = vectorizer.transform(X_dev)
    
    return X_train_counts, X_dev_counts, X_test_counts
    
def tf_idf_rep(X_train_counts, X_dev_counts, X_test_counts):

    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_dev_tf = tf_transformer.transform(X_dev_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)
    
    return X_train_tf, X_dev_tf, X_test_tf


X_train_counts, X_dev_counts, X_test_counts = bow_rep(X_train, X_dev, X_test)

parameters = {'alpha' : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}

nb_classifier = MultinomialNB()
grid_search_clf = GridSearchCV(nb_classifier, parameters, cv=5, return_train_score=True)
grid_search_clf.fit(X_train_counts, y_train)
res = grid_search_clf.cv_results_

for i in range(1, 7):
    print('Rank', i)
    ind = np.where(res['rank_test_score'] == i)
    print('Alpha : {}'.format(res['params'][ind[0][0]]['alpha']))
    print('The mean accuracy : {}\n'.format(round(res['mean_test_score'][ind][0], 3)))
    

# Define the model
nb_classifier = MultinomialNB(alpha=1.)

# Fit the model
nb_classifier.fit(X_train_counts, y_train)


pred_train = nb_classifier.predict(X_train_counts)
pred_dev = nb_classifier.predict(X_dev_counts)
pred_test = nb_classifier.predict(X_test_counts)



print("The accuracy on train set : ", metrics.accuracy_score(y_train, pred_train))

print("The accuracy on dev set : ", metrics.accuracy_score(y_dev, pred_dev))

print("The accuracy on test set : ", metrics.accuracy_score(y_test, pred_test))


print('Classification matrix')
print('------------------ CountVectorizer ------------------\n')
print(classification_report(y_test, pred_test))
print()

# confusion matrix
print('Confusion matrix \n')
print('CountVectorizer confusion matrix')
sns.set(rc={'figure.figsize':(5.7,5.27)})
cm_data = confusion_matrix(y_test, pred_test)
cm = sns.heatmap(cm_data, annot=True, cmap="YlGnBu", linewidths=.5)


X_train_tf, X_dev_tf, X_test_tf = tf_idf_rep(X_train_counts, X_dev_counts, X_test_counts)

parameters = {'alpha' : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}

nb_classifier = MultinomialNB()
grid_search_clf = GridSearchCV(nb_classifier, parameters, cv=5, return_train_score=True)
grid_search_clf.fit(X_train_tf, y_train)
res = grid_search_clf.cv_results_

for i in range(1, 7):
    print('Rank', i)
    ind = np.where(res['rank_test_score'] == i)
    print('Alpha : {}'.format(res['params'][ind[0][0]]['alpha']))
    print('The mean accuracy : {}\n'.format(round(res['mean_test_score'][ind][0], 3)))
    
    
# train the classifier
nb_classifier.fit(X_train_tf, y_train)

pred_train_tf = nb_classifier.predict(X_train_tf)
pred_dev_tf = nb_classifier.predict(X_dev_tf)
pred_test_tf = nb_classifier.predict(X_test_tf)



print("The accuracy on train set : ", metrics.accuracy_score(y_train, pred_train_tf))

print("The accuracy on dev set : ", metrics.accuracy_score(y_dev, pred_dev_tf))

print("The accuracy on test set : ", metrics.accuracy_score(y_test, pred_test_tf))




print('---------------------- TF-IDF -----------------------\n')
print(classification_report(y_test, pred_test_tf))
print()
print('TF-TDF')
sns.set(rc={'figure.figsize':(5.7,5.27)})
cm_data = confusion_matrix(y_test, pred_test_tf)
cm = sns.heatmap(cm_data, annot=True, cmap="YlGnBu", linewidths=.5)


# MLP

parameters = {'alpha' : [.0001, .001, .01, .1, 1.]}

mlp_clf = MLPClassifier()
grid_search_clf = GridSearchCV(mlp_clf, parameters, cv=5, return_train_score=True)
grid_search_clf.fit(X_train_counts, y_train)
res = grid_search_clf.cv_results_

for i in range(1, 6):
    print('Rank', i)
    ind = np.where(res['rank_test_score'] == i)
    print('Alpha : {}'.format(res['params'][ind[0][0]]['alpha']))
    print('The mean accuracy : {}\n'.format(round(res['mean_test_score'][ind][0], 3)))
    

mlp_clf = MLPClassifier(activation='relu', alpha=1., verbose=2, batch_size=50)

# train the classifier
mlp_clf.fit(X_train_counts, y_train)

pred_train_mlp = mlp_clf.predict(X_train_counts)
pred_dev_mlp = mlp_clf.predict(X_dev_counts)
pred_test_mlp = mlp_clf.predict(X_test_counts)



print("The accuracy on train set : ", metrics.accuracy_score(y_train, pred_train_mlp))

print("The accuracy on dev set : ", metrics.accuracy_score(y_dev, pred_dev_mlp))

print("The accuracy on test set : ", metrics.accuracy_score(y_test, pred_test_mlp))


print('---------------------- MLP Classifier with TF-IDF representation -----------------------\n')
print(classification_report(y_test, pred_test_mlp))
print()
print('MLP-TF-TDF')
sns.set(rc={'figure.figsize':(5.7,5.27)})
cm_data = confusion_matrix(y_test, pred_test_mlp)
cm = sns.heatmap(cm_data, annot=True, cmap="YlGnBu", linewidths=.5)



