# YT-Comments-Sentiment
Sentiment analysis of Youtube comment sections using an LSTM (Keras) and the Youtube API. This uses a Long Short Term Memory network (LSTM), a type of recurrent neural network designed to analyze data step-by-step while maintaining previous information long-term. LSTMs are popular in Natural Language Processing as they can retain crucial information such as subjects and context as they recur through sentences. The LSTM is trained on Stanford AI Lab's Large Movie Review Dataset, which contains 50,000 polarized movie reviews labeled as either positive or negative. The trained model is able to achieve ~86% testing accuracy with the movie review dataset. Once the model is trained, comments are scraped via the Youtube Data API v3 from the Youtube video specified by URL. Each comment is analyzed with the LSTM and the overall averages and top positive/negative comments are displayed.

## Installations
Numpy is used for performing various operations on arrays. Matplotlib is used to create the dashboard output. Keras is used to train and load the LSTM. The Google API Client Library and OAuth Library are used to access the Youtube Data API v3. All of these libraries can be installed with the following conda commands:
```
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c anaconda keras
conda install -c conda-forge google-api-python-client
conda install -c conda-forge google-auth-oauthlib
```
## Obtaining Credentials
Youtube's API has a daily quota limit for each account that uses it. This step thus requires a specific Youtube account to access the API. To obtain credentials, you will need to go to the [Google Cloud Console](https://console.cloud.google.com/). From there:
1. Create a new project
2. Enable the Youtube Data API v3 for the new project
3. Go to the OAuth Consent Screen tab on the left and fill out the application with the desired name and email
4. Go to the Credentials tab above and select Create Credentials -> OAuth client ID
5. Select "Other" for application type and create the credentials
6. Download the OAuth client ID as a .json file and rename it "client_secret.json"
7. Place "client_secret.json" in the same directory as this repo

## Running with the pre-built model
main.py takes in two arguments: the Youtube video ID and the model to analyze the comments with. Video IDs can be found in video URLs. To run this with the pre-trained LSTM, titled "sentiment_analysis.h5", enter:
```
py main.py "[insert videoID here]" "sentiment_analysis.h5"
```
If you want to use an alternate model, enter the alternate model's name for the second argument. You may have to change some parameters in the main file, so the comments are preprocessed in the same way as the training data for the alternate model. 
Running main.py will return a dashboard of the comment section sentiments including:
1. **Overall Average Sentiment**, on a scale from 0 (negative) to 1 (positive). Values below 0.4 are considered negative, values between 0.4 and 0.6 inclusive are considered neutral, and values above 0.6 are considered positive. 
1. **Sentiment Proportions** shown in the form of a pie chart, displaying what proportion of the comments were positive, neutral, and negative.
1. **Sentiment Distribution**, shown in the form of a histogram, with each comment counting as a unique data point.
1. **Top 3 Most Positive Comments**, based on the highest calculated sentiment scores, and truncated to 2-3 lines.
1. **Top 3 Most Negative Comments**, based on the lowest calculated sentiment scores, and also truncated to 2-3 lines.

Some interesting examples:

[Bjarne Stroustrup - The Essence of C++](https://www.youtube.com/watch?v=86xWVb4XIyE)





## Creating a different model
The model_creation.py file is provided to document the creation of sentiment_analysis.h5. The model is trained on the Stanford Large Movie Review Dataset, which can be found [here](https://ai.stanford.edu/~amaas/data/sentiment/). If you want to train the model again, you will need to add the /train and /test folders from the dataset to the same directory as this repo. 




 
