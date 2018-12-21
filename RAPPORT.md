### final-lab-RassimAB
final-lab-RassimAB created by GitHub Classroom

# Documents Classification for American Tobacco group

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

