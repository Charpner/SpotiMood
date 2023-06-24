import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import random
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

##########################################################
tracks = pd.read_csv("C:/Users/Cem/Desktop/spotify_dataset/dataset.csv")
print(tracks.head())

##########################################################
print(tracks.shape)

##########################################################
tracks.info()

##########################################################
print(tracks.isnull().sum())

##########################################################
tracks = tracks.drop(['Unnamed: 0', 'track_id'], axis=1)
print(tracks.head())

##########################################################
# Setting implementation of the recommended system by using the most popular 10,000 songs
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)
print(tracks)


x = tracks["danceability"].values
y = tracks["valence"].values

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

fig = plt.figure(figsize=(6, 6))
fig.suptitle("Correlation between danceability and song mood")

ax = plt.subplot(1, 1, 1)
ax.scatter(x, y, alpha=0.5)
ax.plot(x, regr.predict(x), color="red", linewidth=3)
plt.xticks(())
plt.yticks(())

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

plt.xlabel("danceability")
plt.ylabel("valence")

plt.show()

##########################################################
x = "danceability"
y = "valence"

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10, 5))
fig.suptitle("Histograms")
h = ax2.hist2d(tracks[x], tracks[y], bins=20)
ax1.hist(tracks["energy"])

ax2.set_xlabel(x)
ax2.set_ylabel(y)

ax1.set_xlabel("energy")

plt.colorbar(h[3], ax=ax2)

plt.show()

##########################################################
chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]
text1 = tracks["artists"] + " - " + tracks["track_name"]
text2 = text1.values

# X = tracks.drop(droppable, axis=1).values
X = tracks[chosen].values
y = tracks["danceability"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

pca = PCA(n_components=3)
pca.fit(X)

X = pca.transform(X)

# py.init_notebook_mode(connected=True)

trace = go.Scatter3d(
    x=X[:,0],
    y=X[:,1],
    z=X[:,2],
    text=text2,
    mode="markers",
    marker=dict(
        size=4,
        color=y
    )
)

fig = go.Figure(data=[trace])
py.plot(fig, filename="test-graph.html")

##########################################################
chosen = ["energy", "liveness", "tempo", "valence"]
text1 = tracks["artists"] + " - " + tracks["track_name"]
text2 = text1.values

# X = tracks.drop(droppable, axis=1).values
X = tracks[chosen].values
y = tracks["loudness"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)

fig = {
    "data": [
        {
            "x": X[:, 0],
            "y": X[:, 1],
            "text": text2,
            "mode": "markers",
            "marker": {"size": 7, "color": y}
        }
    ],
    "layout": {
        "xaxis": {"title": "How hard is this to dance to?"},
        "yaxis": {"title": "How metal is this?"}
    }
}

py.plot(fig, filename="test-graph1.html")

##########################################################
import time

chosen = ["energy", "liveness", "tempo", "valence", "loudness",
          "speechiness", "acousticness", "danceability", "instrumentalness"]

X = tracks[chosen].values
y = tracks["loudness"].values

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fig = {
    "data": [
        {
            "x": tsne_results[:, 0],
            "y": tsne_results[:, 1],
            "text": text2,
            "mode": "markers",
            "marker": {"size": 8, "color": y}
        }
    ],
    "layout": {
        "xaxis": {"title": "x-tsne"},
        "yaxis": {"title": "y-tsne"}
    }
}

py.plot(fig, filename="test-graph2.html")

##########################################################
tracks['track_name'].nunique(), tracks.shape

##########################################################
tracks = tracks.sort_values(by=['popularity'], ascending=False)
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)

##########################################################
floats = []
for col in tracks.columns:
    if tracks[col].dtype == 'float':
        floats.append(col)
len(floats)

##########################################################
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(2, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
plt.show()

##########################################################
# %%capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['track_genre'])

##########################################################
def recommend(m):
    song_name = ''
    if m.name == 0:                                       #Angry
        song_name = 'Unholy (feat. Kim Petras)'
    if m.name == 1:                                       #Disgust
        song_name = 'you broke me first'
    if m.name == 2:                                       #Fear
        song_name = 'Trampoline (with ZAYN)'
    if m.name == 3:                                       #Happy
        song_name = 'Head & Heart (feat. MNEK)'
    if m.name == 4:                                       #Neutral
        song_name = "I'll Keep You Safe"
    if m.name == 5:                                       #Sad
        song_name = 'Falling'
    if m.name == 6:                                       #Surprise
        song_name = 'La Bachata'

    return song_name

##########################################################
def get_similarities(data, song_name):
    # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data[data['track_name'] == song_name]['track_genre']).toarray()
    num_array1 = data[data['track_name'] == song_name].select_dtypes(include=np.number).to_numpy()

    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['track_name']

        # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data[data['track_name'] == name]['track_genre']).toarray()
        num_array2 = data[data['track_name'] == name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim

##########################################################
def recommend_songs(song_name, data=tracks):
    # Base case

    if tracks[tracks['track_name'] == song_name].shape[0] == -1:
        print('This song is either not so popular or you\
        have entered invalid_name.\n Some songs you may like:\n')

        for song in data.sample(n=5)['track_name'].values:
            print(song)
        return song


    data['similarity_factor'] = get_similarities(data, song_name)

    # First song will be the input song itself as the similarity will be highest.
    data.sort_values(by=['similarity_factor', 'popularity'],
                     ascending=[False, False],
                     inplace=True)

    number1 = random.randint(1, 5717)
    number2 = number1 + 10

    return data[['track_name', 'artists']][number1:number2].to_json(orient='records')


