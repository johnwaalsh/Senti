# YT-Comments-Sentiment
Sentiment analysis of Youtube comment sections using an LSTM and the Youtube API. Uses a Long Short Term Memory network (LSTM) trained on 
Stanford AI Lab's Large Movie Review Dataset, which contains 50,000 polarized movie reviews labeled as either positive or negative. The 
model is able to achieve ~86% testing accuracy with the movie review dataset. Once the model is trained, comments are scraped via the
Youtube Data API v3 from the Youtube video specified by URL. Comments are analyzed with the LSTM and overall averages and top positive/
negative comments are displayed.

## Getting Started
