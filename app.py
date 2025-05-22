import streamlit as st
from scrape_karkidi_jobs import scrape_karkidi_jobs
from preprocess import preprocess_skills, vectorize_skills
from clustering import load_model_and_vectorizer
import joblib

st.title("📢 Job Recommender - Karkidi Scraper")

# Input from user
skills_input = st.text_input("Enter your skills (comma-separated)", "python, machine learning, sql")

if st.button("Find Matching Jobs"):
    with st.spinner("Scraping and analyzing jobs..."):
        df = scrape_karkidi_jobs(keyword="data science", pages=2)
        if df.empty:
            st.warning("No jobs found.")
        else:
            # Preprocess and vectorize
            df = preprocess_skills(df)
            X, _ = vectorize_skills(df)

            # Load trained model and vectorizer
            model = joblib.load("models/kmeans_model.pkl")
            vectorizer = joblib.load("models/vectorizer.pkl")

            # Predict clusters
            df["Cluster"] = model.predict(X)

            # Convert user input to vector
            user_skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
            user_vec = vectorizer.transform([" ".join(user_skills)])
            user_cluster = model.predict(user_vec)[0]

            # Filter matching jobs
            matched_jobs = df[df["Cluster"] == user_cluster]

            st.subheader("🔍 Matching Jobs:")
            if not matched_jobs.empty:
                for _, job in matched_jobs.iterrows():
                    st.markdown(f"""
                    **{job['Title']}** at *{job['Company']}*
                    - 📍 {job['Location']}
                    - 🛠 Skills: {job['Skills']}
                    - 📝 {job['Summary']}
                    ---
                    """)
            else:
                st.info("No matching jobs found.")
