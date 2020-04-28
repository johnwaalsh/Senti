# YT-Comments-Sentiment
Sentiment analysis of Youtube comment sections using an LSTM (Keras) and the Youtube API. This uses a Long Short Term Memory network (LSTM), a type of recurrent neural network designed to analyze data step-by-step while maintaining previous information long-term. LSTMs are popular in Natural Language Processing as they can retain crucial information such as subjects and context as they recur through sentences. The LSTM is trained on Stanford AI Lab's Large Movie Review Dataset, which contains 50,000 polarized movie reviews labeled as either positive or negative. The trained model is able to achieve ~86% testing accuracy with the movie review dataset. Once the model is trained, comments are scraped via the Youtube Data API v3 from the Youtube video specified by URL. Each comment is analyzed with the LSTM and the overall averages and top positive/negative comments are displayed.

## Installations
To run this, you will need to install Keras, the Google api client library, and Matplotlib. This can be done with conda:
```
conda install -c anaconda keras
conda install -c conda-forge google-api-python-client
conda install -c conda-forge google-auth-oauthlib
conda install -c conda-forge matplotlib
```
## Obtaining Credentials
Youtube's API has a daily quota limit for each account that uses it. This step requires a Youtube (Google) account. To obtain credentials, you will need to go to https://console.cloud.google.com/. From there:
1. Create a new project
2. Enable the Youtube Data API v3 for the new project
3. Go to the OAuth Consent Screen tab on the left and fill out the application with the desired name and email
4. Go to the Credentials tab above and select Create Credentials -> OAuth client ID
5. Select "Other" for application type and create the credentials
6. Download the OAuth client ID as a .json file and rename it "client_secret.json"
7. Place "client_secret.json" in the same directory as this repo

## Running with the pre-built model
main.py takes in two arguments: the Youtube video ID and the model to analyze the comments with. To run this with the pre-trained LSTM, enter:
```
py main.py "[insert videoID here]" "sentiment_analysis.h5"
```
Video IDs can be found in video URLs. If you want to use an alternate model, enter the alternate model's name for the second argument (although you may have to change some parameters in the main file). 
Running main.py will return a breakdown of the comment section sentiments:
1. Overall Average Sentiment
2. Top 3 Most Positive Comments
3. Top 3 Most Negative Comments

## Creating a different model
The model_creation.py file is provided to document the creation of sentiment_analysis.h5. The model is trained on the Stanford Large Movie Review Dataset, which can be found at https://ai.stanford.edu/~amaas/data/sentiment/. If you want to train the model again, you will need to add the /train and /test folders from the dataset to the same directory as this repo. 



 
