import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DATA_HEADERS = ['страна', 'рождаемость', 'смертн', 'деск_см', 'длит_муж', 'длит_жен', 'доход', 'регион']

FILE_PATH = 'resource/data.csv'

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 0,
}


# Загрузка данных
def read_data():
    data = pd.read_csv(FILE_PATH, delimiter=';')
    for i in data:
        for j in range(len(data[i])):
            if str(data[i][j]) == 'nan':
                data[i][j] = 0
    print(data[DATA_HEADERS[:1]])
    return data


def hierarchical_clustering(data, normal_data):
    Z = linkage(normal_data, method='ward', metric='euclidean')
    plt.figure(figsize=(25, 15))
    dendrogram(Z, leaf_font_size=15, labels=data[DATA_HEADERS[:1]].values)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()
    return Z


def multidimensional_scaling(data):
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)
    mds = MDS(n_components=2, random_state=0)
    X_mds = mds.fit_transform(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('Multidimensional Scaling')
    plt.show()


def rocky_scree(new_data):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(new_data)
        sse.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()


def normalization(data):
    from sklearn import preprocessing
    return pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data))


def main():
    data = read_data()
    normal_data = normalization(data[DATA_HEADERS[1:]])
    X = hierarchical_clustering(data, normal_data)
    multidimensional_scaling(X)
    rocky_scree(normal_data)


if __name__ == '__main__':
    main()
