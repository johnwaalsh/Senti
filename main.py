import sys
import numpy as np
from string import punctuation
import matplotlib as mpl
import matplotlib.pyplot as plt
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
                    comment = comment.replace("\\n", " ").replace("\\r", " ")
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
def calculate_scores(comments, model_path):
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
    return scores

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
comments = comments = get_video_comments(service, part='snippet', videoId=sys.argv[1], textFormat='plainText', maxResults = 100)
scores = calculate_scores(comments, sys.argv[2])
avg = np.average(scores)
sorted_comments = sorted(zip(scores, comments), reverse=True)

# Calculate the number of comments in each sentiment category
num_pos = sum(1 for i in scores if i > 0.6)
num_neu = sum(1 for i in scores if i <= 0.6 and i >= 0.4)
num_neg = sum(1 for i in scores if i < 0.4)
sizes = [num_pos, num_neu, num_neg]
colors = ["g", "y", "r"]
sizes, colors = (list(l) for l in zip(*sorted(zip(sizes, colors), reverse=True)))

# Auto-indents the displayed comments
def auto_indent(comment, trunc):
    comment = '\n'.join(comment[i:i+70] for i in range(0, len(comment), 70)) 
    comment = (comment[:trunc] + '...') if len(comment) > trunc else comment
    return comment

# Initialize the dashboard display
fig, axs = plt.subplots(3, 2, figsize=(14, 8), facecolor='w')
render = fig.canvas.get_renderer()
pos_box = dict(boxstyle='round', facecolor=(0.62353, 0.89020, 0.67451), alpha=0.5)
neg_box = dict(boxstyle='round', facecolor=(0.89020, 0.59608, 0.59608), alpha=0.5)

# Create a colorbar to represent average sentiment
axs[0, 0].axis('off')
axs[0, 1].axis('off')
cmap = mpl.cm.RdYlGn
top_ax = fig.add_axes([0.32, 0.74, 0.4, 0.05])
sentiment_bar = mpl.colorbar.ColorbarBase(top_ax, cmap=cmap, orientation='horizontal')
sentiment_bar.set_label('Average Sentiment Value : ' + str(round(avg, 5)))
sentiment_bar.set_ticks([])
sentiment_bar.ax.plot([avg, avg], [0, 1], 'w')
top_ax.set_title('Average Sentiment: {}' .format(score_to_sentiment(avg)), fontsize=11)

# Create a pie chart to represent proportions
axs[1, 0].pie(sizes, autopct='%1.1f%%', shadow=False, startangle=90, colors=colors, pctdistance=1.4, textprops={'fontsize': 6})
axs[1, 0].set_title('Sentement Proportions', fontsize=11)

# Create a histogram to represent the distribution
axs[1, 1].hist(scores, 50, facecolor='blue', alpha=0.5)
axs[1, 1].set_xticks([0, 1])
axs[1, 1].set_title('Sentiment Histogram', fontsize = 11)
axs[1, 1].tick_params(axis='both', which='major', labelsize=6)
#axs[1, 1].tick_params(axis='both', which='minor', labelsize=6)

# Set the last two axis to represent top positive and top negative comments
axs[2, 0].axis('off')
axs[2, 0].set_title('Top 3 Positive Comments', fontsize = 11)
axs[2, 1].axis('off')
axs[2, 1].set_title('Top 3 Negative Comments', fontsize = 11)

# Displays the given comment and its sentiment value
def comment(i, bbox, trunc):
    comment = axs[2, 0].text(0, 0, auto_indent("(" + str(round(sorted_comments[i][0][0], 4)) + ") " + sorted_comments[i][1], trunc),
               fontsize=7, ha = 'left', bbox = bbox, verticalalignment = 'top', transform = None)
    height = comment.get_window_extent(renderer=render).height
    return (comment, height)

# Displays comments in the given range
def create_comments(start, stop, bbox, inc):
    comments = []
    for k in range(start, stop, inc):
        if k == stop-inc:
            comments.append(comment(k, bbox, 137))
        else:
            comments.append(comment(k, bbox, 207))
    return comments

# Create the top positive and negative comments
pos_comments = create_comments(0, 3, pos_box, 1)
neg_comments = create_comments(len(sorted_comments) - 1, len(sorted_comments) - 4, neg_box, -1)

# Determine the distance to the bottom of the screen (for positioning comments)
distance_to_bottom = max(sum([h for c, h in pos_comments]), sum([h for c, h in neg_comments])) + 80

# Sets the comments to their correct position depending on screen size
def set_comments(comments, x):   
    comments[0][0].set_position((x, distance_to_bottom))
    comments[1][0].set_position((x, distance_to_bottom - comments[0][1] - 25))
    comments[2][0].set_position((x, distance_to_bottom - comments[0][1] - comments[1][1] - 50))

# Set the comments to their correct position
set_comments(pos_comments, 220)
set_comments(neg_comments, 1040)

# Title with the video ID
fig.suptitle("Sentiment Analysis of \"{}\"".format(sys.argv[1]), y = 0.9, fontsize=18)

# Adjust the bottom and show the data
plt.subplots_adjust(bottom=0.05)
plt.show()

