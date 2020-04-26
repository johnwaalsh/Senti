import os
import random
from collections import Counter
from string import punctuation
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from main import vectorize, word_min_length, max_words as max_words_per_review

# Represents the number of unique words we will use in our counter
words = 15000

# Extracts the text files and respective labels from the given path and shuffles them
def extract_reviews(file):
    file_neg = "{}/neg".format(file)
    file_pos = "{}/pos".format(file)
    neg_X, neg_Y = extract_reviews_per_sentiment(file_neg, 0)
    pos_X, pos_Y = extract_reviews_per_sentiment(file_pos, 1)
    X_total = neg_X + pos_X
    Y_total = neg_Y + pos_Y
    total = list(zip(X_total, Y_total))
    random.shuffle(total)
    X, Y = zip(*total)
    return X, list(Y)
    
# Extracts the text files and respective labels from the given path
def extract_reviews_per_sentiment(file, label):
    X = []
    Y = []
    for i in os.listdir(file):
        f = open("{}/{}" .format(file, i), encoding="utf8")
        text = f.read()
        text = text.translate(str.maketrans('', '', punctuation))
        new_text = []
        for word in text.split():
            if len(word) >= word_min_length:
                word = word.lower()
                new_text.append(word)
        X.append(new_text)
        Y.append(label)
    return X, Y

# Create the lists of text files and labels for training and testing
X_train, Y_train = extract_reviews("train")
X_test, Y_test = extract_reviews("test")

# Create a dictionary for converting the text files to vectors
total_words = [item for sublist in X_train for item in sublist]
word_counts = Counter(total_words)
total_words = sorted(word_counts, key=word_counts.get, reverse = True)
ranking = {}
for num, word in enumerate(total_words, 1):
    if num < words:
        ranking[word] = num
    else:
        break

# Encode the training and testing data
X_train_vectorized = vectorize(X_train, ranking)
X_test_vectorized = vectorize(X_test, ranking)
    
# Pad the reviews with zeros to ensure that they are all the same length (max_words_per_review)
X_train = sequence.pad_sequences(X_train_vectorized, maxlen = max_words_per_review)
X_test = sequence.pad_sequences(X_test_vectorized, maxlen = max_words_per_review)

# Define the model's layers
embedding_size = 32
model = Sequential()
model.add(Embedding(words, embedding_size, input_length = max_words_per_review))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Specify loss and optimization
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

# Split the training data into batches, using the last batch as validation, and train the model
batch_size = 64
num_epochs = 3
X_val, Y_val = X_train[:batch_size], Y_train[:batch_size]
X_training, Y_training = X_train[batch_size:], Y_train[batch_size:]
model.fit(X_training, Y_training, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=num_epochs)

# Judge model accuracy based on the testing data, and save the model
scores = model.evaluate(X_test, Y_test)
print('Test Accuracy:', scores[1])
model.save("sentiment_analysis_5.h5")
np.save("ranking.npy", ranking)


