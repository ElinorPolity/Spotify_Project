import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("SpotifyFeatures.csv")
df.isnull().sum().sum()
df.sample(5)

df.describe()

df.info()

#exploring the data

# Bar Chart of Number of trakcs by genres
artist_count = df.groupby('artist_name')[['genre']].count().reset_index()
artist_count = artist_count.sort_values(by='genre', ascending=True)
artist_count=artist_count.nlargest(25,"genre")

plt.figure(figsize = (36, 6))

sns.barplot(x='artist_name', y='genre', data=artist_count, palette="flare")


plt.xticks(fontsize=14, rotation=40,ha="right")
plt.yticks(fontsize=14)
plt.xlabel('artist', fontsize=18)
plt.ylabel('Number of trucks from a genre', fontsize=18)
plt.title('Number of trakcs by genres for an artist', fontweight='bold', fontsize=22, color='grey')
plt.show()

plt.figure(figsize = (12, 9))
sns.heatmap(df.drop(['genre','artist_name','track_name','track_id','key','time_signature'], axis=1).corr(), annot=True )
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Heatmap', fontsize=22, color='grey')
plt.show()

#According to this heat map, the highest correlation you can see is between loudness and energy (0.82). Also, the popularity has high correlations with loudness (0.36), danceability (0.26), and energy (0.25).

# sum up popularity by genre and divide them by the number of songs in each genre
genre_popularity = df[['genre','track_id', 'popularity']]
genre_count = df.groupby('genre')[['popularity']].count().sort_values(by='popularity', ascending=True)
genre_sum = df.groupby('genre')[['popularity']].sum().sort_values(by='popularity', ascending=True)

genre_rank = genre_sum.popularity / genre_count.popularity
genre_rank = genre_rank.sort_values(ascending=True)
genre_rank = pd.DataFrame(genre_rank).reset_index()

print(genre_rank)

# Bar Chart by category and popularity
plt.figure(figsize = (36, 6))
sns.barplot(x='genre', y='popularity', data=genre_rank,palette="flare")

plt.xticks(fontsize=14, rotation=40,ha="right")
plt.yticks(fontsize=14)
plt.xlabel('Genre', fontsize=18)
plt.ylabel('Popularity', fontsize=18)
plt.title('Average of popularity by genres', fontweight='bold', fontsize=22, color='grey')
plt.show()

#Let’s compare the bar chart of Average of popularity by genres and Number of tracks by genres. Unpopular genres including comedy, soundtrack, indie have higher number of tracks, but in general there’s not quite significant difference in the number of tracks other than A Capella and Children’s music. But there’re two “Children’s music” categories, so if we can just add them up, they have the largest number of tracks.

# Bar Chart of Number of trakcs by genres
genre_count = df.groupby('genre')[['track_id']].count().reset_index()
genre_count = genre_count.sort_values(by='track_id', ascending=True)

plt.figure(figsize = (36, 6))
sns.barplot(x='genre', y='track_id', data=genre_count,palette="flare")

plt.xticks(fontsize=14, rotation=40,ha="right")
plt.yticks(fontsize=14)
plt.xlabel('Genre', fontsize=18)
plt.ylabel('Number of trucks', fontsize=18)
plt.title('Number of trakcs by genres', fontweight='bold', fontsize=22, color='grey')
plt.show()

df_sub2 = df.sample(int(0.017*len(df)))
plt.figure(figsize=(30,15))
num2 = 1

for col in ["acousticness","danceability","duration_ms","energy","instrumentalness","liveness", "popularity", "loudness", "speechiness", "tempo", "valence"]:
    if num2<=10:
        ax = plt.subplot(2,5, num2)
        sns.distplot(a =df_sub2[col])
        plt.xlabel(col,fontsize = 17)
    num2 +=1
plt.suptitle("Distribution of Musical Attributes",fontsize = 23)
plt.show()

# **------------------------------------------------**

#part 2 - **clusters**

data = pd.read_csv("SpotifyFeatures.csv")

indx = data[['track_name', 'artist_name']]
attributes = data.drop(['track_id', 'time_signature','track_name', 'artist_name', 'key'], axis = 1)
attributes.head()

print(data.shape)
print(data['genre'].value_counts(normalize=True))
data_samples = data.sample(n=100000)
print(data_samples.shape)
print(data_samples['genre'].value_counts(normalize=True))

ordinal_encoder = OrdinalEncoder()
object_cols = ['mode']
attributes[object_cols] = ordinal_encoder.fit_transform(attributes[object_cols])

attributes = pd.get_dummies(attributes)
attributes.insert(loc=0, column='track_name', value=indx.track_name)
attributes.insert(loc=1, column = 'artist_name', value = indx.artist_name)

genres_names = ['genre_A Capella', 'genre_Alternative', 'genre_Anime', 'genre_Blues',
       "genre_Children's Music", "genre_Children’s Music", 'genre_Classical',
       'genre_Comedy', 'genre_Country', 'genre_Dance', 'genre_Electronic',
       'genre_Folk', 'genre_Hip-Hop', 'genre_Indie', 'genre_Jazz',
       'genre_Movie', 'genre_Opera', 'genre_Pop', 'genre_R&B', 'genre_Rap',
       'genre_Reggae', 'genre_Reggaeton', 'genre_Rock', 'genre_Ska',
       'genre_Soul', 'genre_Soundtrack', 'genre_World']
      
      genres = attributes.groupby(['track_name', 'artist_name'])[genres_names].sum()

column_names = ['track_name', 'artist_name']
for i in genres_names:
    column_names.append(i)

genres.reset_index(inplace=True)
genres.columns = column_names

attributes = attributes.drop(genres_names, axis = 1)

atts_cols = attributes.drop(['track_name', 'artist_name'], axis = 1).columns
scaler = StandardScaler()
attributes[atts_cols] = scaler.fit_transform(attributes[atts_cols])

songs = pd.merge(genres, attributes, how = 'inner', on = ['track_name', "artist_name"])
songs = songs.drop_duplicates(['track_name', 'artist_name']).reset_index(drop = True)

songs.head()

#**PART 2.2 KMEAN cluster**
#Exploratory Data Analysis - for choosing K

sse={}
DF = pd.DataFrame(songs.drop(['track_name', 'artist_name'], axis = 1))
for k in range(1, 30,3):
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(DF)
    DF["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.title("Elbow method")
plt.xlabel("Number of cluster")
plt.show()

#K-means with Cosine Distance model

DF = pd.DataFrame(songs.drop(['track_name', 'artist_name'], axis = 1))
kmeans = KMeans(n_clusters=17)
songs['Cluster'] = kmeans.fit_predict(DF)

#**part 2.3 - EM**

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=39, init_params='random')
model.fit(DF)
songs['Cluster']=model.predict(DF)

#**part 2.4 - DBSCAN**

from sklearn.cluster import DBSCAN
model_DB= DBSCAN(eps=3, min_samples=100)
model_DB.fit(DF[:100000])
#songs[0:10000]['cluster']=model_DB.labels_

**PART 3 - try to visualize **

# Visualizing the Clusters with t-SNE
# Visualizing the Clusters with t-SNE
from sklearn.manifold import TSNE

def visual(X):

  tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
  genre_embedding = tsne_pipeline.fit_transform(X)
  projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)

  projection['genre'] = genre_data['genre']
  projection['cluster'] = genre_data['cluster']

  val = np.array(genre_data['cluster'])
  for i in range(4000):
    projection.at[i, "cluster"] = val[i]

  val = np.array(genre_data['genre'])
  for i in range(4000):
    projection.at[i, "genre"] = val[i]
    
  fig = px.scatter(
      projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genre'])
  fig.show()

#k-mean

df_copy=pd.read_csv("SpotifyFeatures.csv")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
genre_data = df_copy.sample(n=4000)
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=17))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)
visual(X)

 #GaussianMixture
 
df_copy=pd.read_csv("SpotifyFeatures.csv")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
genre_data = df_copy.sample(n=4000)
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('GaussianMixture', GaussianMixture(n_components=39, init_params='random'))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)
visual(X)

# 'DBSCAN'

df_copy=pd.read_csv("SpotifyFeatures.csv")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
genre_data = df_copy.sample(n=4000)
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('DBSCAN', DBSCAN(eps=3, min_samples=100))])
X = genre_data.select_dtypes(np.number)
genre_data['cluster'] =cluster_pipeline.fit(X)
#genre_data['cluster'] = cluster_pipeline.predict(X)
visual(X)
