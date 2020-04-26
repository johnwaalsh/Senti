import numpy as np
from string import punctuation
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from keras.models import load_model
from keras.preprocessing import sequence

# Parameters set for the pre-trained model, do not change these unless you create 
# a new model
word_min_length = 3
max_words = 1500

# This file is specific to each individual user, must be downloaded from Google
# Cloud
client_secret_file = "client_secret.json"

# API specifics
scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
api_service_name = 'youtube'
api_version = 'v3'

# Build the service using the user's credentials
app_flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
credentials = app_flow.run_console()
service = build(api_service_name, api_version, credentials = credentials)

# The maximum number of comments on a video, should be set to correspond with the
# Youtube API's daily quota (10,000 units for most acounts)
max_num_comments = 7500

# Retrieves all of the comments from the specified video
def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
    total_comments = 0
    
    while total_comments < (max_num_comments / 100):
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            replies = service.comments().list(part = 'snippet', parentId = item['id']).execute()
            if len(replies['items']) != 0:
                for reply in replies['items']:
                    comment = reply['snippet']['textDisplay']
                    comment = comment.replace("&#39;", "'")
                    comment = comment.replace("\n", " ")
                    comments.append(comment)
        
        total_comments += 100
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break
 
    return comments

# Loads the pre-trained ranking dictionary
ranking = np.load('ranking.npy').item()

# Calculates the prediction scores for the comments on the given video ID using
# the given model path
def calculate_scores(video_id, model_path):
    comments = get_video_comments(service, part='snippet', videoId=video_id, textFormat='plainText', maxResults = 100)
    X_data = []
    for c in comments:
        new_text = []
        c = c.translate(str.maketrans('', '', punctuation))
        for word in c.split():
            if len(word) >= word_min_length:
                word = word.lower()
                new_text.append(word)
        X_data.append(new_text)
        
    X_data_vectorized = vectorize(X_data, ranking)
    X_data_vectorized = sequence.pad_sequences(X_data_vectorized, maxlen = max_words)
    
    model = load_model(model_path)
    scores = model.predict(X_data_vectorized)
    avg = np.average(scores)
    print("Average Sentiment : " + str(score_to_sentiment(avg)))
    print("Average Sentiment Value: " + str(avg))
    
    sorted_comments = sorted(zip(scores, comments), reverse=True)
    print("Top 3 Positive Comments: ")
    print("1. {} : {}\n2. {} : {}\n3. {} : {}".format(sorted_comments[0][0], sorted_comments[0][1], sorted_comments[1][0], 
                                         sorted_comments[1][1], sorted_comments[2][0], sorted_comments[2][1])) 
    print("Top 3 Negative Comments: ") 
    print("1. {} : {}\n2. {} : {}\n3. {} : {}".format(sorted_comments[len(sorted_comments)-1][0], 
                                                  sorted_comments[len(sorted_comments)-1][1], 
                                                  sorted_comments[len(sorted_comments)-2][0], 
                                                  sorted_comments[len(sorted_comments)-2][1], 
                                                  sorted_comments[len(sorted_comments)-3][0], 
                                                  sorted_comments[len(sorted_comments)-3][1]))

# Encodes the given text files
def vectorize(text_data, ranking):
    text_data_vectorized = []
    for review in text_data:
        review_vectorized = []
        for word in review:
            rank = ranking.get(word, 0)
            if rank != 0:
                review_vectorized.append(rank)
        text_data_vectorized.append(review_vectorized)
    return text_data_vectorized

# Converts the given score to a sentiment
def score_to_sentiment(i):
    if i > 0.6:
        return "Positive"
    elif i < 0.4:
        return "Negative"
    else:
        return "Neutral"

# Runs for the given video ID and model ("sentiment_analysis.h5" is pre-trained)
calculate_scores(sys.argv[1], sys.argv[2])