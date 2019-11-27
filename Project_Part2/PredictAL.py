#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:15:42 2018

@author: katezeng

This module is for Predictive Analysis - Hypothesis Testing
    - This component contains both the traditional statistical hypothesis testing, and the beginning of machine learning predictive analytics. 
      Here you will write three (3) hypotheses and see whether or not they are supported by your data. You must use all of the methods listed below 
      (at least once) on your data.
    - You do not need to try all the methods for each hypothesis. For example, you might use ANOVA for one of your hypotheses, and you might use a 
      t-test and linear regression for another, etc. It will be the case, that some of the hypotheses will not be well supported.
    - When trying methods like a decision tree, you should use cross-validation and show your ROC curve and a confusion matrix. For each method, 
      explain the method in one paragraph.
    - Explain how and why you will apply your selected method(s) to each hypothesis, and discuss the results.
    - Therefore, you will have at least three (3) hypothesis tests and will apply all seven (7) of the following methods to one or more of your 
      hypotheses.
    - Required methods:
        - t-test or Anova  (choose one)
        - Linear Regression or Logistical Regression  (multivariate or multinomial) (choose one)
        - Decision tree
        - A Lazy Learner Method (such as kNN)
        - Naïve Bayes
        - SVM
        - Random Forest
"""

#####################################################
#                                                   #
#                Import Libraries                   #
#                                                   #
#####################################################
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


########################################################
#                                                      #
#                 List of Functions                    #
#                                                      #
########################################################
# function for arranging columns
def arrangeCol(data):
    cols = list(data)
    cols.insert(len(cols), cols.pop(cols.index('price')))
    data = data.loc[:, cols]
    return data

# function for linear regression with absolute error plot
def linearRegression1(data):
    X = data[['hotel_meanprice']]
    y = data[['price']]
    X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    plt.figure(figsize=(15,8))
    ax = sns.distplot(y_test-predictions)
    ax.set(ylabel='Density', xlabel='Error', 
           title='Error distribution of test sets by Linear Regrssion model')
    plt.savefig("./plots/LRresults.png")


# function for linear regression with absolute error vs actual value
def linearRegression2(data):
    X = data[['hotel_meanprice']]
    y = data[['price']]
    X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    plt.figure(figsize=(15,8))
    ax = sns.distplot(abs(y_test-predictions)/y_test)
    ax.set(ylabel='Percentage', xlabel='Mean Squared Error', 
           title='Error distribution of test sets by Linear Regrssion model')
    plt.savefig("./plots/LR_absolute_diff.png")

# find relationship between hotel average price and airbnb average price
def hotel_airbnb(data):
    output1 = data.groupby(['zipcode'])['price'].mean().reset_index()
    output1.columns = ['zipcode', 'averagePrice']
    output2 = data.groupby(['zipcode'])['hotel_meanprice'].mean().reset_index()
    output = pd.merge(output1, output2, on='zipcode')
    plt.figure(figsize=(15,8))
    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(output['hotel_meanprice'], output['averagePrice'])
    ax = sns.regplot(x='hotel_meanprice', y='averagePrice', data=output, 
                     line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
    ax.set(xlabel='Hotel prices', ylabel='Airbnb prices', 
           title='Linear relationship between average hotel prices and Airbnb prices')
    ax.legend()
    plt.savefig("./plots/relationship_hotel_airbnb.png")

# find the distribution of airbnb price
def find_distribution(data):
    plt.figure(figsize=(15,8))
    ax = sns.distplot(data['price'])
    ax.set(ylabel='density', xlabel='Airbnb Prices', title='Airbnb Price Distribution')
    plt.savefig("./plots/airbnb_price_dist.png")

# find the impact of room type by doing one way ANOVA
def room_type_impact(data):
    entire_apt = np.array(data[data['Entire home/apt'] == 1]['price'])
    shared_room = np.array(data[data['Shared room'] == 1]['price'])
    private_room = np.array(data[data['Private room'] == 1]['price'])
    result = stats.f_oneway(entire_apt, private_room, shared_room)
    print(result)

# preproccessing data for further model training
def preprocessing(data):
    price_dict = {'A': 0, 'B': 1, 'C': 2}
    data['price_group'] = pd.cut(data.price, bins=[0, 200, 400, 1000], labels=[0, 1, 2])
    cols = ['latitude', 'longitude', 'zipcode', 'price']
    data = data.drop(cols, axis=1)
    mydict = {'t': 1, 'f': 0}
    data = data.replace({'host_profile_pic': mydict})
    data = data.replace({'identity_verified': mydict})
    
    fig = plt.figure(figsize=(10, 8))
    data.groupby('price_group').price_group.count().plot.bar(ylim=0)
    fig.suptitle('class distribution', fontsize=15)
    plt.xlabel('price group', fontsize=12)
    plt.xticks(rotation='horizontal')
    plt.ylabel('Number of hotels', fontsize=12)
    fig.savefig('./plots/class_distribution.jpg')
    
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.DataFrame(data.iloc[:, -1])
    y = y.values.ravel()
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    col_names = data.columns.tolist()
    new_data = np.c_[X_res, np.transpose(y_res)]
    data = pd.DataFrame(new_data, columns = col_names)
    
    return data, price_dict

# function for model evaluation including classification report, accuracy score
# and generate confusion matrix of each model
def model_evaluation(y_test, y_pred, name):
    ## for confusion matrix
    # class info
    class_names = ['A', 'B', 'C'] 

    conf_mat = confusion_matrix(y_test, y_pred)
    
    print("========Confusion Matrix and Reprot of " + name + "==========")

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    #sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_yticklabels(), rotation=45)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    #plt.savefig('./plots/confusion-matrix' + name + '.png')
    
    ## for accuracy score
    print("Accuracy Score of " + name + "\n", accuracy_score(y_test, y_pred))
    
    ## for classification report
    print("Classification Report of " + name + "\n", classification_report(y_test, y_pred))


# training and testing using naive bayes classifier, and generate ROC curve
def naiveBayes(data):
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.factorize(data['price_group'])[0]
    norm = Normalizer()
    X = norm.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    
    # define classifier
    clf = GaussianNB()
    
    # ROC curve
    y_score = cross_val_predict(clf, X, y, cv=10 ,method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using Naive Bayes Classifier')
    plt.legend(loc="lower right")
    plt.savefig('./plots/naive_bayes_roc.png')
    plt.show()
    
    # model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    
    # make predictions for test data and evaluate
    pred_y = clf.predict(X_test)
    
    return y_test, pred_y

# function for training and testing using decision tree classifier, and generate ROC curve
def decisionTree(data):
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.factorize(data['price_group'])[0]
    norm = Normalizer()
    X = norm.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    
    # define classifier
    clf = DecisionTreeClassifier()
    
    # ROC curve
    y_score = cross_val_predict(clf, X, y, cv=10 ,method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using Decision Tree Classifier')
    plt.legend(loc="lower right")
    plt.savefig('./plots/decision_tree_roc.png')
    plt.show()
    
    # model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    
    # make predictions for test data and evaluate
    pred_y = clf.predict(X_test)
    
    return y_test, pred_y

# function for training and testing using KNN classifier, and generate ROC curve
def KNN_classifier(data):
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.factorize(data['price_group'])[0]
    norm = Normalizer()
    X = norm.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    
    # define classifier
    clf = KNeighborsClassifier(n_neighbors = 5)
    
    # ROC curve
    y_score = cross_val_predict(clf, X, y, cv=10 ,method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using KNN Classifier')
    plt.legend(loc="lower right")
    plt.savefig('./plots/KNN_roc.png')
    plt.show()
    
    # model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    
    # make predictions for test data and evaluate
    pred_y = clf.predict(X_test)
    
    return y_test, pred_y

# function for training and testing using svm classifier, and generate ROC curve
def svm_classifier(data):
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.factorize(data['price_group'])[0]
    norm = Normalizer()
    X = norm.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    
    # define classifier
    clf = svm.SVC(gamma='auto', kernel='rbf',probability = True)
    
    # ROC curve
    y_score = cross_val_predict(clf, X, y, cv=10 ,method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using support vector machine')
    plt.legend(loc="lower right")
    plt.savefig('./plots/svm_roc.png')
    plt.show()
    
    # model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    
    # make predictions for test data and evaluate
    pred_y = clf.predict(X_test)
    
    return y_test, pred_y

# function for training and testing using random forest classifier, 
# generate ROC curve, and feature importance
def random_forest(data):
    X = pd.DataFrame(data.iloc[:, 0:-1])
    y = pd.factorize(data['price_group'])[0]
    norm = Normalizer()
    X = norm.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Binarize the output
    y_bin = label_binarize(y, classes=[0, 1, 2])
    #n_classes = y_bin.shape[1]
    
    # define classifier
    clf = RandomForestClassifier(n_estimators=100)
    
    # ROC curve
    y_score = cross_val_predict(clf, X, y, cv=10 ,method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data using random forest')
    plt.legend(loc="lower right")
    plt.savefig('./plots/random_forest_roc.png')
    plt.show()
    
    # model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    feat_labels = data.columns.tolist()
    feature_importance = list(zip(feat_labels, clf.feature_importances_))
    feature_importance = pd.DataFrame(feature_importance, columns = ['feature', 'importance'])
    feature_importance = feature_importance.sort_values(by = ['importance'], ascending = False)
    feature_importance.to_csv('./results/feature_importance_full.csv', index=False)
    
    # make predictions for test data and evaluate
    pred_y = clf.predict(X_test)
    predictions = [np.round(value) for value in pred_y]
    total_accuracy = accuracy_score(y_test, predictions)
    print("RFC Accuracy: %.2f%%" % (total_accuracy * 100.0))
    
    feats = {} # a dict to hold feature_name: feature_importance
    selectnumber = 20
    outcome = pd.read_csv('./results/feature_importance_full.csv')
    outcome = outcome['feature'][0:selectnumber]

    for feature, importance in zip(outcome[0:selectnumber], sorted(clf.feature_importances_[0:selectnumber],reverse = True)):
        feats[feature] = importance #add the name/value pair

    impo_plot = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    #impo_plot = impo_plot.sort_values(by='Gini-importance')

    impo_plot.sort_values(by='Gini-importance').plot(figsize=(18, 6), kind='bar').invert_xaxis()

    plt.savefig('./plots/gini_importance_selected.png')
    
    return y_test, pred_y

# generate report for each methods
def generate_report(names, test_list, result_list):
    for i in range(len(names)):
        model_evaluation(test_list[i], result_list[i], names[i])


########################################################
#                                                      #
#              Function Call and Results               #
#                                                      #
########################################################
        
def main():
    # import data
    mydata = pd.read_csv("./data/Airbnb_Cleaned.csv")
    mydata = mydata[(mydata.price <= 1000) & (mydata.price != 0)]
    #mydata.head(10)
    #mydata.info()

    # call the list of predefined functions
    mydata = arrangeCol(mydata)

    linearRegression1(mydata)

    linearRegression2(mydata)

    hotel_airbnb(mydata)

    find_distribution(mydata)

    room_type_impact(mydata)
    
    # print to view some info of data
    print("Mean price for airbnb: ", np.mean(mydata['price']))
    print("Max price for airbnb: ", max(mydata['price']))
    print("Min price for airbnb: ", min(mydata['price']))

    predic_data, price_dict = preprocessing(mydata)

    names = ['Naive Bayes', 'Decission Tree', 'KNN', 'SVM', 'Random Forest']
    test_list = []
    result_list = []

    nb_test, nb_pred = naiveBayes(predic_data)
    test_list.append(nb_test)
    result_list.append(nb_pred)

    dt_test, dt_pred = decisionTree(predic_data)
    test_list.append(dt_test)
    result_list.append(dt_pred)

    knn_test, knn_pred = KNN_classifier(predic_data)
    test_list.append(knn_test)
    result_list.append(knn_pred)

    svm_test, svm_pred = svm_classifier(predic_data)
    test_list.append(svm_test)
    result_list.append(svm_pred)

    rf_test, rf_pred = random_forest(predic_data)
    test_list.append(rf_test)
    result_list.append(rf_pred)

    generate_report(names, test_list, result_list)

if __name__ == "__main__":
    main()