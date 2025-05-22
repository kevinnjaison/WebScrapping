import streamlit as st
from scrape_karkidi_jobs import scrape_karkidi_jobs
from preprocess import preprocess_skills, vectorize_skills
from clustering import cluster_jobs
import pandas as pd

st.title("📢 Job Notifier - Karkidi Scraper")

skills_input = st.text_input("Enter your skills (comma-separated)", "python, machine learning, sql")

if st.button("Find Matching Jobs"):
    with st.spinner("🔍 Scraping and analyzing jobs..."):
        df = scrape_karkidi_jobs(keyword="data science", pages=2)

        if df.empty:
            st.warning("No jobs found.")
        else:
            # Clean and vectorize
            df = preprocess_skills(df)
            X, vectorizer = vectorize_skills(df)

            # Fresh KMeans model every time
            model = cluster_jobs(X, n_clusters=5)
            df["Cluster"] = model.labels_

            # Match jobs to user skills
            user_skills = [skill.strip().lower() for skill in skills_input.split(",") if skill.strip()]
            matched_jobs = []

            for i, row in df.iterrows():
                job_skills = row["Skills"].lower().split(",")
                job_skills = [skill.strip() for skill in job_skills]
                match_score = len(set(job_skills) & set(user_skills))
                if match_score > 0:
                    matched_jobs.append(row)

            # Show results
            if matched_jobs:
                st.success(f"Found {len(matched_jobs)} matching jobs:")
                for job in matched_jobs:
                    st.markdown(f"""
                    ### {job['Title']} at {job['Company']}
                    📍 {job['Location']}  
                    🛠 Skills: {job['Skills']}  
                    📝 {job['Summary']}  
                    ---  
                    """)
            else:
                st.info("No matching jobs found.")
