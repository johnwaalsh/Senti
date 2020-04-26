import numpy as np
from string import punctuation
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from keras.models import load_model
from keras.preprocessing import sequence
from model_creation import word_min_length, max_words, vectorize

client_secret_file = "client_secret.json"

scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
api_service_name = 'youtube'
api_version = 'v3'

app_flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
credentials = app_flow.run_console()
service = build(api_service_name, api_version, credentials = credentials)

max_num_comments = 7500

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
        
    X_data_vectorized = vectorize(X_data)
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

def score_to_sentiment(i):
    if i > 0.6:
        return "Positive"
    elif i < 0.4:
        return "Negative"
    else:
        return "Neutral"
    
calculate_scores(sys.argv[0], sys.argv[1])
    

    

