from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import spotipy

from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline

from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist



data = pd.read_csv("data.csv")

sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

X = data.select_dtypes(np.number)
number_cols = list(X.columns)

song_cluster_pipeline = pickle.load(open('model.pkl','rb'))

song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

# Created a developer account for spotify web api. there we can get unique client_id and client_secret
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='0ce040c2e8034283ac4b22c9eea94216',client_secret='2255beebeb454222b25d6a1aaf724850'))

# finding song using the spotify api which returns the data abt the song(valence, acoustiness, etc..)
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        # first try to find the song data in the dataset 
        # if not found in the dataset this code will throw and error 
        # and we will get the data from the spotify API
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        # Creating a song_vector which contains all the attributes found for the song
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=6):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    #get center of song cluster
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)


    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    song_title = request.form['songTitle']
    year = request.form['year']
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@',song_title,year)
    output = recommend_songs([{'name': song_title, 'year': int(year)}],data)
    # print('Output: :::::', output)
    master = '''<div class="center"><h2>Recommendations</h2></div>'''
    html_str_1 = '''<div class="container p-0 mt-2 bg-dark text-white"><p class="centerY">'''
    html_str_2 = '''</p><p class="centerZ">'''
    html_str_3 = '''</p></div>'''
    
    for i in output:
        temp = html_str_1 + i['name'] + ' - '+i['artists'] + html_str_2 + str(i['year'])+html_str_3
        master+=temp
    return render_template('index.html', prediction_text = master+temp)



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    # data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    # output = prediction[0]
    # return jsonify(output)
    # song_title = request.POST.get['songTitle']
    data = request.POST.get('data')
    song_title = data['songTitle']
    year = data['year']
    # print("-----data----- ",data)

    # first try with one song then implement may songs

    output = recommend_songs([{'name': song_title, 'year': int(year)}],data)  
    # print(output)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
