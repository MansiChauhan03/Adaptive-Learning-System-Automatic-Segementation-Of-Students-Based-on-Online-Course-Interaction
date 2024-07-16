importing all the necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
reading the dataset

student_data=pd.read_csv('/content/Health_Science_Student_Engagement_Grade_Data.csv')

from google.colab import drive
drive.mount('/content/drive')

checking the first five rows along with coloumn names
student_data.head()

Returning the number of rows and columns of the DataFrame
student_data.shape

Printing information about the DataFrame
student_data.info()

student_data.isnull().sum()

list of features to be used from the dataset
features = ['assignment_submission', 'assignment_view', 'resource_view', 'course_access', 'quiz', 'discussion_access', 'Engagement']

scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(student_data[features])

sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

creating a elbow plot for finding optimum number of clusters
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

performing k means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
student_data['Cluster'] = kmeans.fit_predict(data_scaled)

computing the mean values of specified features for each cluster
cluster_summary = student_data.groupby('Cluster')[features].mean().reset_index()
print(cluster_summary)
utilizing sns to create a pair plot for visualizing relationships between variables in student_data, differentiated by clusters.
Two variables are examined simultaneously in order to look for patterns, dependencies, or interactions between them.

sns.pairplot(student_data, hue='Cluster', vars=features, palette='viridis', plot_kws={'alpha':0.7})
plt.suptitle('Cluster Visualization Based on Online Course Interaction', y=1.02)
plt.show()
 calculating the average engagement level for each cluster in dataset.

cluster_engagement = student_data.groupby('Cluster')['Engagement'].mean()
plt.figure(figsize=(8, 6))
plt.bar(cluster_engagement.index, cluster_engagement.values, color='skyblue', edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Average Engagement')
plt.title('Average Engagement by Cluster')
plt.show()

using Principal Component Analysis from scikit-learn to reduce the dimensionality of data_scaled to two principal components (n_components=2)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
plt.figure(figsize=(10, 8))

visualizing the results of Principal Component Analysis by creating a scatter plot of the data points in the reduced two-dimensional space, colored and labeled according to their clusters
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.scatter(data_pca[student_data['Cluster'] == i][:, 0],
                data_pca[student_data['Cluster'] == i][:, 1],
                s=50,
                c=colors[i],
                label=f'Cluster {i+1}')

plt.title('PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.show()
