# Documents Classification for American Tobacco group

##### Rassim ABDOU

## Introduction

  In this project we are interessted in a database that contain different type of documents in an image and text file formats collected from five big Americans tobacco groups, so the U.S. government sued them for amassing large benefit by lying about the dangers of cigarettes. 
  
  The goal of this project is to make an automatic classification of Documents types in order to facilitate the exploitation of these documents by the lawyers.
  
  So first we will undertake and analysis of the data present in our database, then we define a process for creating the traning, validation, and test bases so we can propose different models and test different classification algorithmes, and finally we will discuss about the results of these models and future works.

## 1 - Dataset description and analysis

  The Database contains 3482 text documents distributed in several categories listed below :
  
  * Advertisement.
  * Email.
  * Form.
  * Letter.
  * Memo.
  * News.
  * Note.
  * Report.
  * Resume.
  * Scientific.
  
    We can see the distribution of the documents over these 10 classes in the following image :
  ![Distribution of the documents](https://github.com/Rouen-NLP/final-lab-RassimAB/blob/master/classesCount.png)
  
  According to this plot, the distribution is not very unbalanced besides the number of Memos, Emails and Letters Which are slightly raised in this dataset.
  
## 2 - Problem Analysis
  
  In this section, we'll analyse the problem and discuss about the solutions that may solve it with the best performances.

  ### 2.1 - Train, validation & test data 
  
  While we don't have a large number of documents, the validation and test sizes should not be very low which takes us to devide the dataset as follow : 
  
  Train | Validation | Test
------------ | ------------- | -------------
70% | 15% | 15%

  ```
  # Splitting the data
  X_train, X_test, y_train, y_test = train_test_split(df[df.columns[0]], df[df.columns[1]], test_size=0.30)
  X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.50)
  ```

  ### 2.2 - Data representations : (BoW & TF-IDF)
  
   We'll use two different text representations which are Bag of words and TF-IDF. Then compare how the models that we'll define deal with those sturctures and which ones gives the best performances with the best hyperparameters. 
   
  ```
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
  ```
  
  Now to choose the best machine learning model, we'll make several test and compare how these 2 data representations infuence the performances. First we set up a naïve bayes model which is the most commun and one of the best models for text classification because it's based on probabilities, and try to figure out which best hyperparameters it would fit with, then we set up a neural network (MLP).
  
## 3 - Documents classification & performance analysis 

  ### 3.1 - Naïve Bayes classifier
  
  - The idea is based on running multiple Naïve bayes classifiers with several values of the hyperparameter 'alpha', starting from without smoothing (alpha = 0) to 5.0 with a step or 1. And apply a 5-fold cross validation to test the model accuracy by computer the mean of the 5 experiments
  - We'll start with the BoW representation and then TF-IDF.
  
  The following are the results with BoW representation :
  
  Rank | alpha | Acc
------------ | ------------- | -------------
1 | 1.0 | 68,8%
2 | 2.0 | 68,5%
3 | 3.0 | 67,9%
4 | 4.0 | 67,6%
5 | 5.0 | 66,9%
6 | 0.0 | 63,2%

  Results with TF-IDF representation :
  
   Rank | alpha | Acc
------------ | ------------- | -------------
1 | 1.0 | 62,5%
2 | 0.0 | 62,2%
3 | 2.0 | 57,2%
4 | 3.0 | 53,3%
5 | 4.0 | 50,3%
6 | 5.0 | 48,3%

-  According to the grid search approach, the default value of alpha (1.0) with BoW representation gives the best performances and so we'll build a real Naïve bayes model and compute its accuracy on the test data then discuss the precisions and recalls along with the confusion matrix.


Data | Acc 
------------ | ------------- 
Train | 80,5% 
Dev | 71,3% 
Test | 71,8% 

Confusion matrix : 

  ![Distribution of the documents](https://github.com/Rouen-NLP/final-lab-RassimAB/blob/master/cm_nb.png)

  After analysing the confusion matrix, it shows that the model gives good performances for classifing emails and resumes with 94% and 95% precisions respectively as well as Forms and Scientific documents. And the global accuracy was arround 71% as mentioned in the table above. 
  
  We deduce that a Naïve bayes classifier is pretty performant in term of text classification, now lets try another model based on neural networks which is MLP.


  ### 3.2 - MLP classifier
  
  - The MLP classifier have been trained and tested with also the BoW representation while it's the representation that fits the best for these data. For the hyperparameters we've done the same approach as before to get the best value for the parameter that influence the performance of the MLP (which is alpha), the results we got are the following :

Rank | alpha | Acc
------------ | ------------- | -------------
1 | 1.0 | 74,1%
2 | 0.001 | 73,7%
3 | 0.0001 | 73,6%
4 | 0.01 | 73,4%
5 | 0.1 | 72,6%

  - According to these approach, the value of the hyperparameter alpha that gives the best performances (1.0) have been used to train the model on the whole training set with 17 iterations, and then applied on the validation and test sets, the results are the following: 

Data | Acc 
------------ | ------------- 
Train | 94,7% 
Dev | 71,3% 
Test | 76%

Confusion matrix : 

![Distribution of the documents](https://github.com/Rouen-NLP/final-lab-RassimAB/blob/master/cm_mlp.png)

  - The confusion matrix is well-balanced comparing the the previous one, there are even some classes that has almost 100% of score (like Resume and Email) and the classification error is well distributed over all the classes which means that the classifier knows how to generalize.

  - If we want to compare all the models with every data representation that we did above, we deduce that the mlp with BoW representation gives the best performances comparing to naïve bayes model, even if this last is slightly different, and about the tf-idf representations, it wasn't a good method to classify this kind of documents.

## 4 - Conclusion and future works

  - To conclude, we've saw that text classification is a complex problem, and solving it depands on the data and its representation in the preprocessing step, and of course on the model. 
  
  - Here we've improved the performances from a classifier to another, and we still can improve it if we dive into Recurrent neural networks which are the state-of-the-arts now for text classifications, one way to do that if instead of using the representations above, we can use the data in a sequential way to feed into to LSTMs layers or even Biderectional LSTMs which gives good performances in this kind of problems.
