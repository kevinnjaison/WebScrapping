import streamlit as st
from scraper import scrape_karkidi_jobs
from preprocess import preprocess_skills, vectorize_skills
from clustering import cluster_jobs
from notifier import notify_users  # Will extend to email
import os

st.title("📢 Job Notifier - Karkidi Scraper")

skills_input = st.text_input("Enter your skills (comma-separated)", "python, machine learning, sql")

if st.button("Find Matching Jobs"):
    with st.spinner("Scraping and analyzing jobs..."):
        df = scrape_karkidi_jobs(keyword="data science", pages=2)
        if df.empty:
            st.warning("No jobs found.")
        else:
            df = preprocess_skills(df)
            X, _ = vectorize_skills(df)
            cluster_jobs(X)

            user_skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
            st.subheader("🔍 Matching Jobs:")
            matched_jobs = notify_users(df, user_skills, return_results=True)

            if matched_jobs:
                for job in matched_jobs:
                    st.markdown(f"""
                    **{job['Title']}** at *{job['Company']}*
                    - 📍 {job['Location']}
                    - 🛠 Skills: {job['Skills']}
                    - 📝 {job['Summary']}
                    ---
                    """)
            else:
                st.info("No matching jobs found.")
