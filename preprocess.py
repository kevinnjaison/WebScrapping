import pandas as pd
import re

def preprocess_skills(df):
    df["Skills"] = df["Skills"].fillna("").apply(lambda x: re.sub(r'[\n\r]', ', ', x))
    return df

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["Skills"])
    return X, vectorizer
