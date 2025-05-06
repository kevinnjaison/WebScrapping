def notify_users(df, user_skills: list, return_results=False):
    matched_jobs = []

    for skill in user_skills:
        matches = df[df['Skills'].str.contains(skill, case=False, na=False)]
        matched_jobs.extend(matches.to_dict('records'))

    unique_jobs = {(j['Title'], j['Company']): j for j in matched_jobs}.values()

    if return_results:
        return list(unique_jobs)

    for job in unique_jobs:
        print(f"🔔 {job['Title']} at {job['Company']}")
        print(f"📍 {job['Location']}")
        print(f"🛠 Skills: {job['Skills']}")
        print(f"📝 Summary: {job['Summary']}")
        print("-" * 50)

    return []
