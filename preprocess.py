import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_skills(df):
    df = df.copy()
    df['Skills'] = df['Skills'].str.lower().str.replace(',', ' ').str.replace('-', ' ')
    return df

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])
    return X, vectorizer
