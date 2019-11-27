# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:02:11 2018

@author: Prerna Kaul
"""
import pandas as pd
from sklearn import decomposition 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


# Transform Categorical variables into dummies:     
def Category_to_Dummy(data,category_var_list):
    for i in category_var_list:
        # Create dummies variables and concatenate them
        dummies=pd.get_dummies(data[i])
        data=pd.concat([data,dummies],axis=1)
        
        # Drop the categorical variable
        data=data.drop([i],axis=1)
        
    return (data)


# Function to Perform KMeans Clustering
def kmeans(normalizedDataFrame):
    
    print("\n**********K-Means Clustering*********\n")
    
    # testing which cluster is best fit
    for i in range(2,11):
        
        kmeans = KMeans(n_clusters=i)
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        
        #determining if the clustering is good
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        print("\n\nFor number of clusters =", i , "The average silhouette_score is :", silhouette_avg)
    
        # converting high dimensioanl data into 2D
        pca2D = decomposition.PCA(2)

        # Turn the data into two columns with PCA
        plot_columns = pca2D.fit_transform(normalizedDataFrame)

        # Plot using a scatter plot and shade by cluster label
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
    
    
# =============================================================================
#      From observations of the graphs and silhouette score, 
#       the best number of clusters = 3    
# =============================================================================
    print("\n\nBest K-Means clustering:")
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("\nFor number of clusters =", 3 , "The average silhouette_score is :", silhouette_avg)
    
    # converting high dimensioanl data into 2D
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)

    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("K-Means Clustering")
    plt.show()        
    
  

# Function to Perform Agglomerative Ward Clustering
def ward(normalizedDataFrame):
    
    print("\n\n**********Agglomerative Clustering*********\n")
    
    # testing which cluster is best fit
    for i in range(2,11):
        
        al = AgglomerativeClustering(n_clusters = i)
        cluster_labels = al.fit_predict(normalizedDataFrame)
    
        #determining if the clustering is good
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        print("\n\nFor number of clusters =" , i , "The average silhouette_score is :", silhouette_avg)
    
        # converting high dimensioanl data into 2D
        pca2D = decomposition.PCA(2)

        # Turn the data into two columns with PCA
        plot_columns = pca2D.fit_transform(normalizedDataFrame)

        # Plot using a scatter plot and shade by cluster label
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
        
        
# =============================================================================
#     From observations of the graphs and the silhouette score, 
#       the best number of clusters = 3        
# =============================================================================
    print("\n\nBest Agglomerative Ward clustering:")
    al = AgglomerativeClustering(n_clusters = 3)
    cluster_labels = al.fit_predict(normalizedDataFrame)
    
    #determining if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("\nFor number of clusters =" , 3 , "The average silhouette_score is :", silhouette_avg)
    
    # converting high dimensioanl data into 2D
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)

    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("Agglomerative Ward Clustering")
    plt.show()
    
    

# Function to Perform DBSCAN Clustering
def db(normalizedDataFrame):
    
    print("\n\n**********DBSCAN Clustering*********\n")
    
    # testing which cluster is best fit
    ep = [0.5,0.6,0.7,0.8,0.9,1]
    for i in range(len(ep)):
        
        dbs = DBSCAN(eps = ep[i], min_samples = 200)
        cluster_labels = dbs.fit_predict(normalizedDataFrame)
    
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print('\nEstimated number of clusters: %d' % (n_clusters+1))
    
        #determining if the clustering is good
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        print("The average silhouette_score is for the estimated number of clusters is :", silhouette_avg)
    
        # converting high dimensioanl data into 2D
        pca2D = decomposition.PCA(2)

        # Turn the data into two columns with PCA
        plot_columns = pca2D.fit_transform(normalizedDataFrame)

        # Plot using a scatter plot and shade by cluster label
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
    
    
# =============================================================================
#     From observations of the graphs and the silhouette score, 
#       the best number of clusters = 4, when eps = 0.6        
# =============================================================================
    print("\n\nBest DBSCAN clustering:")
    dbs = DBSCAN(eps = 0.6, min_samples = 200)
    cluster_labels = dbs.fit_predict(normalizedDataFrame)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print('\nEstimated number of clusters: %d' % (n_clusters+1))
    
    #determining if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("The average silhouette_score is for the estimated number of clusters is :", silhouette_avg)
    
    # converting high dimensioanl data into 2D
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)

    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("DBSCAN Clustering")
    plt.show()
  
    
    
def main():
    df = pd.read_csv("./data/Airbnb_Cleaned.csv")
    
    category_var_list = {'property_type','room_type','bed_type'}
    df = Category_to_Dummy(df,category_var_list)
    
    df = df.iloc[1:10000:, 1:55] # removing the index column
    
    # removing variables that do not contribute to cluster analysis
    df = df.drop(["host_profile_pic", "identity_verified"], axis = 1) 
    #print(df.head())
    
    # Preparing data for clustering
    x = df.values #returns a numpy array
    # mormalising the data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    kmeans(normalizedDataFrame)
    
    ward(normalizedDataFrame)
    
    db(normalizedDataFrame)
    
    
main()