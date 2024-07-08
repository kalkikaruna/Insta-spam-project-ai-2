import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import logging
import os
import socket
import hashlib
import threading
import time

# Set up logging
logging.basicConfig(filename='instagram_spam_detector.log', level=logging.INFO)

def detect_spam(text):
    logging.info("Detecting spam in text")
    # Preprocess the text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove usernames
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    text = ' '.join(tokens)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    text_tfidf = vectorizer.transform([text])

    # Train a Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(text_tfidf, [0])  # Assume the text is not spam

    # Evaluate the Naive Bayes model
    y_pred_nb = nb_model.predict(text_tfidf)
    logging.info("Naive Bayes prediction:", y_pred_nb)

    # Train a deep learning model
    embedding_dim = 128
    max_length = 200
    pca_dim = 50

    # Create a PCA model
    pca = PCA(n_components=pca_dim)
    text_pca = pca.transform(text_tfidf.toarray())

    # Create a deep learning model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_length))
    model.add(LSTM(64, dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the deep learning model
    model.fit(text_pca, [0], epochs=10, batch_size=32)

    # Evaluate the deep learning model
    y_pred_dl = model.predict(text_pca)
    y_pred_dl = np.argmax(y_pred_dl, axis=1)
    logging.info("Deep Learning prediction:", y_pred_dl)

    # Use the best model to make a prediction
    if y_pred_dl == 1:
        logging.info("Spam detected!")
        return True
    else:
        logging.info("Not spam")
        return False

def block_spam_traffic():
    logging.info("Blocking spam traffic")
    # Block traffic to known spam servers
    block_list = ["example.com", "badguy.net"]
    for host in block_list:
        os.system(f"iptables -A OUTPUT -d {host} -j DROP")

    # Block suspicious traffic patterns
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 80))
    sock.listen(1)
    conn, addr = sock.accept()
    data = conn.recv(1024)
    if b"GET /spam" in data:
        logging.info("Blocking suspicious traffic from", addr)
        conn.close()

def incident_response():
    logging.info("Incident response activated")
    # Contain the incident
    os.system("iptables -A INPUT -s 0.0.0.0/0 -j DROP")

    # Notify the incident response team
    logging.info("Spam incident detected, notifying team...")

def monitor_system():
    logging.info("Monitoring system for spam activity")
    while True:
        # Check for spam activity
        detect_spam("This is a test message")
        time.sleep(60)  # Check every 60 seconds

def main():
    logging.info("Starting Instagram spam detector")
    # Create a thread to monitor the system
    thread = threading.Thread(target=monitor_system)
    thread.daemon = True
    thread.start()

    # Run the incident response script
    incident_response()

if __name__ == "__main__":
    main
