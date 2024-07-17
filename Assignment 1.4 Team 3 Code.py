import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.decomposition import PCA

tracks_df = pd.read_csv("spotify_tracks.csv")
tracks_df.head()

#removed duplicates
tracks_df = tracks_df.groupby(['track_name', 'artists']).first().reset_index()
tracks_df.shape

#histogram 
tracks_df.hist(bins = 30, figsize=(20,15))
plt.suptitle("Histograms for each variables")
plt.show()

#heatmap
numeric_df = tracks_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre'])
corr = numeric_df.corr()

#strong relationship between variables
plt.scatter(x='loudness', y='energy', data=tracks_df, s=0.5) 
plt.suptitle("Scatter plot of the relationship between Loudness and Energy")
plt.xlabel('Loudness') 
plt.ylabel('Energy') 
plt.show() 

plt.scatter(x='energy', y='acousticness', data=tracks_df, s=0.5) 
plt.suptitle("Scatter plot of the relationship between Energy and Acousticness")
plt.xlabel('Energy') 
plt.ylabel('Acousticness') 
plt.show()

plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

#Part 2:





#To avoid TypeErrors
numeric_columns = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
                   'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo',               
       'time_signature', 'valence']  # Adjust this list based on your actual numeric columns

numeric_df = tracks_df[numeric_columns]
numeric_df = numeric_df.dropna(subset=numeric_columns)

# Z-score method to get rid of outliers
def RemoveOutliersZScore(df, threshold=3):
    z_scores = stats.zscore(df)
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    return df[filtered_entries]

# Applying method to df
Trimmed_df = RemoveOutliersZScore(numeric_df)

# Showing before and after of df
print(f"Original df:\n{numeric_df.shape}")
print(f"Updated df:\n{Trimmed_df.shape}")

# Elbow Method
def ElbowMethod(df):
    Sum_Of_Distance = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df)
        Sum_Of_Distance.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), Sum_Of_Distance, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    plt.show()

# Showing the optimal amount of clusters
ElbowMethod(Trimmed_df)

subsampled_df = Trimmed_df.sample(frac=0.005, random_state=42)

# Apply PCA to reduce the number of dimensions to 2
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(subsampled_df)

# KMeans Clustering

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(Trimmed_df)

# Ensure indices and lengths match before assignment (added for Part 3)
if len(kmeans.labels_) == len(Trimmed_df.index):
    tracks_df.loc[Trimmed_df.index, 'kmeans_cluster'] = kmeans.labels_
    # Fill NaN values with a placeholder (e.g., -1) before converting to integers
    tracks_df['kmeans_cluster'].fillna(-1, inplace=True)
    tracks_df['kmeans_cluster'] = tracks_df['kmeans_cluster'].astype(int)  # Ensure the cluster labels are integers
else:
    print("Error: Mismatch in length of indices and KMeans labels.")

# Print unique values in kmeans_cluster to verify (added for Part 3)
print(tracks_df['kmeans_cluster'].unique())

#Function to plot clusters for reusability

def plot_clusters(data, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='energy', y='acousticness', hue=labels, palette='viridis', data=data, s=50)
    plt.title(title)
    plt.xlabel('Energy')
    plt.ylabel('Acousticness')
    plt.legend(loc='best')
    plt.show()
    
def plot_clusters_pca(data, reduced_data, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', s=50)
    plt.title(title)
    plt.xlabel('Energy')
    plt.ylabel('Acousticness')
    plt.legend(loc='best')
    plt.show()

#Plotting Kmeans clustering
plot_clusters(tracks_df, 'kmeans_cluster', 'KMeans Clusters')

#DBSCAN Clustering
dbscan = DBSCAN(eps=0.1, min_samples=15)
labels = dbscan.fit_predict(reduced_data)

# Add the labels to the df
subsampled_df = subsampled_df.reset_index()
subsampled_df['dbscan_cluster'] = labels

plot_clusters_pca(subsampled_df, reduced_data, labels, 'DBSCAN Clustering with Adjusted Parameters')

#Agglomerative Clustering
Agglomerative = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels = Agglomerative.fit_predict(reduced_data)

#Labels
subsampled_df = subsampled_df.reset_index()
subsampled_df['Agglomerative_Cluster'] = labels


#Plotting Agglomerative Clustering
plot_clusters_pca(subsampled_df, reduced_data, labels, 'Agglomerative Clustering with PCA Components')


#Part 3:

# Only numeric columns are used
numeric_df_with_clusters = tracks_df[numeric_columns + ['kmeans_cluster']]

# Calculate the correlation matrix
corr = numeric_df_with_clusters.corr()

# Check if all clusters are included
clusters_present = tracks_df['kmeans_cluster'].unique()
for cluster in clusters_present:
    if cluster not in corr.index:
        corr.loc[cluster] = [0] * len(corr.columns)
        corr[cluster] = 0

# Print the correlation matrix
print(corr)

# Two valid song ids for testing purposes
valid_ids = tracks_df.loc[Trimmed_df.index, 'track_id'].head(2).tolist()
print("Valid song IDs for testing:", valid_ids)

# Prompt the user to enter their favorite songs' IDs (ensure they are string-based)
ids = input('Enter comma-separated IDs of your favorite songs:\n> ').strip().split(',')

# Trim whitespace from each ID
ids = [id.strip() for id in ids]

# Filter the dataframe to get the user's favorite songs
favorites = tracks_df[tracks_df['track_id'].isin(ids)]

# Find most frequent cluster
clusters = favorites['kmeans_cluster'].value_counts()
user_favorite_cluster = clusters.idxmax()

print('\nFavorite cluster:', user_favorite_cluster, '\n')

# Suggest songs
suggestions = tracks_df[tracks_df['kmeans_cluster'] == user_favorite_cluster]


print("Top 5 song suggestions:")
print(suggestions[['track_name', 'artists', 'album_name']].head())

# Function to get recommendations from different clusters
def get_different_recommendations(tracks, favorite_cluster, corr):
    if favorite_cluster in corr.index:
        different_clusters = corr.loc[favorite_cluster].sort_values(ascending=True).index.tolist()
        different_clusters.remove(favorite_cluster)
        
        different_suggestions = tracks[tracks['kmeans_cluster'].isin(different_clusters)].head(10)
        return different_suggestions
    else:
        print(f"Cluster {favorite_cluster} not found in the correlation matrix.")
        return pd.DataFrame()  # Return an empty DataFrame if the cluster is not found

# Collect feedback
def collect_feedback(tracks, user_favorite_cluster, corr):
    feedback = input("\nDid you like these recommendations? (yes/no):\n> ").strip().lower()
    if feedback == 'yes':
        print("\nGreat!")
    elif feedback == 'no':
        print("\nSorry to hear that. Try these songs:")
        different_suggestions = get_different_recommendations(tracks, user_favorite_cluster, corr)
        if not different_suggestions.empty:
            print(different_suggestions[['track_name', 'artists', 'album_name']])
        else:
            print("No alternative recommendations available.")
    else:
        print("\nInvalid input. Please enter 'yes' or 'no'.")

# Use the feedback function
collect_feedback(tracks_df, user_favorite_cluster, corr)

# Ensure the data is numeric
numeric_columns = ['acousticness', 'danceability', 'duration_ms', 'energy',
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity',
                  'speechiness', 'tempo', 'time_signature', 'valence']  # Adjust this list based on your actual numeric columns


numeric_df = tracks_df[numeric_columns]


# Handle NaN
numeric_df = numeric_df.dropna(subset=numeric_columns)




# Z-score method to get rid of outliers
def RemoveOutliersZScore(df, threshold=3):
   z_scores = stats.zscore(df)
   abs_z_scores = abs(z_scores)
   filtered_entries = (abs_z_scores < threshold).all(axis=1)
   return df[filtered_entries]




# Applying method to df
Trimmed_df = RemoveOutliersZScore(numeric_df)


# Showing before and after of df
print(f"Original df:\n{numeric_df.shape}")
print(f"Updated df:\n{Trimmed_df.shape}")


# ELBOW Method
def ElbowMethod(df):
   Sum_Of_Distance = []
   for i in range(1, 11):
       kmeans = KMeans(n_clusters=i, random_state=42)
       kmeans.fit(df)
       Sum_Of_Distance.append(kmeans.inertia_)
   plt.figure(figsize=(10, 6))
   plt.plot(range(1, 11), Sum_Of_Distance, marker='o')
   plt.title('Elbow Method')
   plt.xlabel('Number of clusters')
   plt.ylabel('Error')
   plt.show()


#Functions to plot clusters for reusability
def plot_clusters(data, labels, title):
   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='energy', y='acousticness', hue=labels, palette='viridis', data=data, s=50)
   plt.title(title)
   plt.xlabel('Energy')
   plt.ylabel('Acousticness')
   plt.legend(loc='best')
   plt.show()


# Optimal amount of clusters
ElbowMethod(Trimmed_df)


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(Trimmed_df)


# Ensure indices and lengths match before predictions
if len(kmeans.labels_) == len(Trimmed_df.index):
   tracks_df.loc[Trimmed_df.index, 'kmeans_cluster'] = kmeans.labels_
   tracks_df['kmeans_cluster'] = tracks_df['kmeans_cluster'].fillna(-1)
   tracks_df['kmeans_cluster'] = tracks_df['kmeans_cluster'].astype(int)  # Ensure the cluster labels are integers
else:
   print("Error: Mismatch in length of indices and KMeans labels.")
plot_clusters(tracks_df, 'kmeans_cluster', 'KMeans Clusters')


# Ensure only numeric columns are used for correlation calculation
numeric_df_with_clusters = tracks_df[numeric_columns + ['kmeans_cluster']]


# Calculate and plot the correlation matrix
corr = numeric_df_with_clusters.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
plt.title('Data with Clusters Correlation Matrix Heatmap')
# plt.show()


# Remove duplicate songs
tracks_df = tracks_df.drop_duplicates(subset=["artists", "track_name"])


# Prompt favorite songs' names and artists(ensure they are string-based and not case sensitive)
songs = input('Enter comma-separated song titles and artists of your favorite songs: (e.g: Let It Be - Remastered 2015:The Beatles)\n> ').strip().split(',')
songs = [song.strip().lower().split(':') for song in songs]
song_names = []
song_artists = []
for i in range(len(songs)):
   for j in range(2):
       songs[i][j] = songs[i][j].strip()
   song_names.append(songs[i][0])
   song_artists.append(songs[i][1])


favorites = tracks_df[tracks_df['track_name'].str.lower().str.strip().isin(song_names) &
                     tracks_df['artists'].str.lower().str.strip().isin(song_artists)]


# Find most frequent cluster
clusters = favorites['kmeans_cluster'].value_counts()
user_favorite_cluster = clusters.idxmax()


print('\nFavorite cluster:', user_favorite_cluster, '\n')


# Suggest songs
suggestions = tracks_df[tracks_df['kmeans_cluster'] == user_favorite_cluster]


#Function to collect user's mood and generate range
def ask_mood():
   mood = input('How would you rate your mood today on a scale of 1-5, where 1 is the worst and 5 is the best?\n>')
   mood_range = [0, 0]
   mood=int(mood)
   if(mood not in range(1,6)):
       print("Invalid input, try again:\n>")
   else:
       mood = int(mood)
       mood_range[0] = (mood-1)*(0.6)
       mood_range[1] = mood_range[0]+0.6
   return mood_range


#Ask a Y/N question for a situation
def ask_situation(question, yes='Y', no='N'):
   situation = input(question)
   if(situation==yes):
       return True
   elif(situation==no):
       return False
   else:
       print("Invalid input, try again:")
       return ask_situation(question)


#Filter for specific situations
def filter_situation():
   track_sf = suggestions
   wanted = ask_situation("Do you want further filtering? Y/N\n>")
   if (not wanted):
       return track_sf
   pg = ask_situation("Are you with young children/family? Y/N\n>")
   explicity = lambda x : track_sf[track_sf['explicit']==False] if x else track_sf
   track_sf = explicity(pg)
   sing = ask_situation("Are you looking for songs to sing along to? Y/N\n>")
   singalong = lambda x : track_sf[track_sf['popularity']>=75] if x else track_sf
   track_sf = singalong(sing)
   dance = ask_situation("What about dance? Y/N\n>")
   dancealong = lambda x : track_sf[track_sf['danceability']>0.8] if x else track_sf
   track_sf = dancealong(dance)
   background = ask_situation("Are you looking for background music, or lyrical music? B/L\n>", 'B', 'L')
   instrumental = lambda x : track_sf[track_sf['speechiness']<0.2] if x else track_sf[track_sf['speechiness']>0.2]
   track_sf = instrumental(background)
   return track_sf


mood_range = ask_mood()


sugggestions = suggestions[(suggestions['valence']+suggestions['energy']+suggestions['liveness']>mood_range[0]) &
                          (suggestions['valence']+suggestions['energy']+suggestions['liveness']<mood_range[1])]


suggestions = filter_situation()
# Sort by popularity
suggestions = suggestions.sort_values(by='popularity', ascending=False)


print("Top 5 song suggestions:")
print(suggestions[['track_name', 'artists', 'album_name']].head())




# Function to get recommendations from different clusters
def get_different_recommendations(tracks, favorite_cluster, corr):
   if favorite_cluster in corr.index:
       different_clusters = corr.loc[favorite_cluster].sort_values(ascending=True).index.tolist()
       different_clusters.remove(favorite_cluster)


       different_suggestions = tracks[tracks['kmeans_cluster'].isin(different_clusters)].head(10)
       return different_suggestions
   else:
       print(f"Cluster {favorite_cluster} not found in the correlation matrix.")
       return pd.DataFrame()  # Return an empty DataFrame if the cluster is not found




# Feedback
def collect_feedback(tracks, user_favorite_cluster, corr):
   feedback = ask_situation("Did you like these recommendations? Y/N \n")
   if feedback:
       print("\nGreat!")
   elif not feedback:
       print("\nSorry to hear that. Try these songs:")
       different_suggestions = get_different_recommendations(tracks, user_favorite_cluster, corr)
       if not different_suggestions.empty:
           print(different_suggestions[['track_name', 'artists', 'album_name']])
       else:
           print("No alternative recommendations available.")
   else:
       print("\nInvalid input. Please enter 'yes' or 'no'.")




# Prompt feedback
collect_feedback(tracks_df, user_favorite_cluster, corr)


# Collect initial feedback
collect_feedback()
