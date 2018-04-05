# load packages
import os
import pandas as pd
import nltk
import gensim
import pickle
import re

# read data
os.chdir('C:\\Users\Lala No.5\Desktop\Final_Thesis')
df = pd.read_csv("train_clean.csv")

# transform dataframe to list
description = df.item_description.values.tolist()
# read stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))


# Function to filter stop words from tokens for each sentence
def sentence_filter(sentence_token, stop_words):
    filtered = []
    for token in sentence_token:
        if not token in stop_words:
            filtered.append(token)
    return filtered

# Clean the sentence
def sentence_cleaner(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

# Input description, tokenize each sentence and
# return token for each sentence without stopping words
def sentence_tokenizer(description, stop_words):
    value = []
    for sentence in description:
        sentence = sentence.lower()
        sentence = sentence_cleaner(sentence)
        sentence_token = nltk.word_tokenize(sentence)
        filtered_token = sentence_filter(sentence_token,
                                         stop_words)
        value.append(filtered_token)
    return value


# Tokenize the sentence
description_token = sentence_tokenizer(description, stop_words)

# Skip gram model with minimum word count = 10
# and output vector of length 300
model_sg_1 = gensim.models.Word2Vec(description_token,
                                    min_count=10, size=300, sg=1)

# Skip gram model with minimum word count = 50
# and output vector of length 300
model_sg_2 = gensim.models.Word2Vec(description_token,
                                    min_count=50, size=300, sg=1)

# CBOW(continuous bag of words) model
model_cbow = gensim.models.Word2Vec(description_token,
                                    min_count=10, size=300)

# save and reload
# For tokens of description
with open("decription_token.txt", "wb") as fp:  # Pickling
    pickle.dump(description_token, fp)

with open("decription_token.txt", "rb") as fp:  # Unpickling
    description_token = pickle.load(fp)

# For both models
model_sg_1.save('model_sg_1')
model_sg_2.save('model_sg_2')

