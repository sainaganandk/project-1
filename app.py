import pandas as pd
import numpy as np
import pickle
from flask import Flask,render_template,Request,request
import pip
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from textstat import flesch_reading_ease
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from textblob import TextBlob
import spacy
import string
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
# Download required NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
    # Load the spaCy model
nltk.download('vader_lexicon')
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('wordnet')
from collections import Counter
import math


def average_sentence_length(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Calculate the length of each sentence and sum them up
    total_sentence_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    # Calculate the average sentence length
    average_length = total_sentence_length / len(sentences)
    return average_length
def get_vocabulary_richness(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    vocab_richness = len(set(filtered_words)) / len(filtered_words)
    return vocab_richness

def get_readability_score(text):
    readability_score = flesch_reading_ease(text)
    return readability_score
def get_punctuation_usage(text):
    punctuation_counts = Counter(char for char in text if char in string.punctuation)
    return punctuation_counts
def get_sentiment_scores(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores
def gunning_fog(text):
    import re
    sentences = re.split(r'[.!?]', text)
    total_words = 0
    complex_words = 0
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                syllables = count_syllables(word)
                if syllables >= 3:
                    complex_words += 1
    avg_words_per_sentence = total_words / len(sentences)
    percentage_complex_words = (complex_words / total_words) * 100
    fog_index = 0.4 * (avg_words_per_sentence + percentage_complex_words)
    return fog_index
def count_syllables(word):
    count = 0
    vowels = "aeiouy"
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
def per(t):
    import nltk
  #nltk.download('punkt')
    from nltk.lm.preprocessing import padded_everygram_pipeline
    from nltk.lm import Laplace
    from nltk.tokenize import word_tokenize
      # Tokenize the text
    tokens = word_tokenize(t)
    # Create bigrams
    n = 3
    train_data, padded_vocab = padded_everygram_pipeline(n, tokens)
    # Train a Laplace-smoothed model
    model = Laplace(n)
    model.fit(train_data, padded_vocab)
    # Calculate perplexity on the same text
    test_data = list(nltk.ngrams(tokens, n))
    perplexity = model.perplexity(test_data)
    return perplexity
def preprocess_text(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Tokenize each sentence into words, remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []
    for sentence in sentences:
        words = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
        preprocessed_sentences.append(" ".join(words))
    return preprocessed_sentences
def compute_coherence(text):
    preprocessed_sentences = preprocess_text(text)
    # Convert preprocessed sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    # Compute cosine similarity between sentence vectors
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Average pairwise similarities to get coherence score
    num_sentences = len(preprocessed_sentences)
    total_similarity = sum(similarity_matrix[i][j] for i in range(num_sentences) for j in range(num_sentences) if i != j)
    coherence_score = total_similarity / (num_sentences * (num_sentences - 1))
    return coherence_score
def shannon_entropy(text):
    import math
    # Count occurrences of each character
    frequencies = Counter(text)
    # Total number of characters in the text
    total_chars = len(text)
    # Calculate probability of each character
    probabilities = [freq / total_chars for freq in frequencies.values()]
    # Calculate Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy
Model = tf.keras.models.load_model('my_model.h5')


def fun(test):
    x_validate=pd.DataFrame()
    x_validate=pd.DataFrame()
    g_average_sentence_length=[]
    g_get_vocabulary_richness=[]
    g_compute_coherence=[]
    g_gunning_fog=[]
    g_per=[]
    g_shannon_entropy=[]
    try:
        g_average_sentence_length.append(average_sentence_length(test))
    except:
        g_average_sentence_length.append(average_sentence_length(0))
    try:
        g_get_vocabulary_richness.append(get_vocabulary_richness(test))
    except:
        g_get_vocabulary_richness.append(get_vocabulary_richness(0))
    try:
        g_compute_coherence.append(compute_coherence(test))
    except:
        g_compute_coherence.append(compute_coherence(0))
    try:
        g_gunning_fog.append(gunning_fog(test))
    except:
        g_gunning_fog.append(gunning_fog(0))
    try:
        g_per.append(per(test))
    except:
        g_per.append(per(0))
    try:
        g_shannon_entropy.append(shannon_entropy(test))
    except:
        g_shannon_entropy.append(shannon_entropy(0))
    x_validate['average_sentence_length']=g_average_sentence_length
    x_validate['get_vocabulary_richness']=g_get_vocabulary_richness
    x_validate['compute_coherence']=g_compute_coherence
    x_validate['gunning_fog']=g_gunning_fog
    x_validate['per']=g_per
    x_validate['shannon_entropy']=g_shannon_entropy
    Ans=Model.predict(x_validate)
    list_=[]
    list_.append(Ans)
    list_.append(g_average_sentence_length)
    list_.append(g_get_vocabulary_richness)
    list_.append(g_compute_coherence)
    list_.append(g_gunning_fog)
    list_.append(g_per)
    list_.append(g_shannon_entropy)
    return list_




app=Flask(__name__)

@app.route("/")


def home():
    return render_template('main.html')

@app.route('/project',methods=["POST"]) 

def project():
    Text=request.form['textInput']
    ans=fun(Text)
    processed_text_1 = ans[1]
    processed_text_2 = ans[2]
    processed_text_3 = ans[3]
    processed_text_4 = ans[4]
    processed_text_5 = ans[5]
    processed_text_6 = ans[6]
    if ans[0]== [0]:
        return render_template('human.html', processed_text_1=processed_text_1, processed_text_2=processed_text_2, processed_text_3=processed_text_3,
                               processed_text_4=processed_text_4, processed_text_5=processed_text_5, processed_text_6=processed_text_6)
    else:
        return render_template('ai.html',processed_text_1=processed_text_1, processed_text_2=processed_text_2, processed_text_3=processed_text_3,
                               processed_text_4=processed_text_4, processed_text_5=processed_text_5, processed_text_6=processed_text_6)
    

