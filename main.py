from scrape_karkidi_jobs import scrape_karkidi_jobs
from preprocess import preprocess_skills, vectorize_skills
from clustering import cluster_jobs, save_model
from notifier import notify_users
import os

def main():
    print("🔎 Scraping job listings...")
    df = scrape_karkidi_jobs(keyword="data science", pages=2)
    
    if df.empty:
        print("No jobs scraped. Exiting.")
        return

    print("🧹 Preprocessing data...")
    df = preprocess_skills(df)

    print("🔠 Vectorizing skills...")
    X, vectorizer = vectorize_skills(df)

    print("🤖 Clustering jobs...")
    model = cluster_jobs(X, n_clusters=5)

    print("💾 Saving model and vectorizer...")
    os.makedirs("models", exist_ok=True)
    save_model(model, vectorizer)

    # ✅ Define user skill preferences
    user_skills = ["python", "machine learning", "nlp"]

    # ✅ Notify based on the jobs just scraped
    notify_users(df, user_skills)

if __name__ == "__main__":
    main()
