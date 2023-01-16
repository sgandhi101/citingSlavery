import nltk
import pandas as pd
from sklearn.metrics import euclidean_distances
from wordcloud import WordCloud

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Read court cases from HCLAP
court_cases = pd.read_csv("slavery_citing_cases.csv")

text_list = []
for case_body in court_cases["casebody"]:
    start = case_body.find("text':")
    if start >= 0:
        start += len("text':")  # move the index past "text':"
        end = case_body.find("'", start)  # look for next ' after text':
        if end > start:
            text_list.append(case_body[start:end])

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

preprocessed_cases = []
for case in text_list:
    # tokenize case text
    words = word_tokenize(case)
    # remove stop words
    words = [word for word in words if word.lower() not in stop_words]
    # stem words
    words = [stemmer.stem(word) for word in words]
    # add preprocessed case text to new list
    preprocessed_cases.append(words)

from gensim.models import Word2Vec

# train word2vec embeddings on preprocessed cases
model = Word2Vec(preprocessed_cases, window=5, min_count=1, workers=4)

from sklearn.cluster import KMeans

# create a word-vectors array
word_vectors = model.wv.vectors

# cluster word vectors using k-means
kmeans = KMeans(n_clusters=10, random_state=0).fit(word_vectors)

# get the cluster assignments for each word
clusters = kmeans.labels_.tolist()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# create a dictionary to map words to their cluster
word_clusters = dict(zip(model.wv.index_to_key, clusters))


def label_points(x, y, val, ax):
    for i, xi in enumerate(x):
        ax.annotate(val[i], (xi, y[i]), size=10, textcoords='offset points', ha='center', va='center')


# project the word vectors to 2D space
tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(word_vectors)

# plot the words on a 2D plane
plt.scatter(Y[:, 0], Y[:, 1], c=clusters)

plt.show()

# Get the top 10 closest words for each cluster center
top_n = 10
all_closest_words_indices = []
for i in range(10):
    distances = euclidean_distances(word_vectors, kmeans.cluster_centers_[i].reshape(1, -1))
    closest_words_indices = distances.argsort()[:, :top_n]
    all_closest_words_indices.append(closest_words_indices)
    if closest_words_indices.shape[1] > 0:
        closest_words = [model.wv.index_to_key[index] for index in closest_words_indices[0]]
        print("Cluster %d: %s" % (i, ', '.join(closest_words)))
    else:
        print(f"Cluster {i}: Empty list")

# Create word clouds for each cluster
for i in range(10):
    closest_words_indices = all_closest_words_indices[i]
    if closest_words_indices.shape[1] > 0:
        cluster_words = [model.wv.index_to_key[index] for index in closest_words_indices[0]]
        # Create a wordcloud for the cluster
        wordcloud = WordCloud(background_color="white", max_words=200, contour_color='steelblue',
                              contour_width=3).generate(' '.join(cluster_words))
    else:
        print(f"Cluster {i} not enough elements to create wordcloud")
