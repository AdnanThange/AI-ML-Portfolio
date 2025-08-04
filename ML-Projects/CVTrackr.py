!pip install pymupdf scikit-learn nltk pandas openpyxl

from google.colab import files
import fitz
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

uploaded = files.upload()
pdf_filenames = list(uploaded.keys())

def extract_text_from_pdf(filename):
    text = ""
    with fitz.open(filename) as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

use_case = input("Enter your use-case: ")
use_case_clean = preprocess(use_case)

skills_input = input("Enter skill(s) to extract (comma-separated): ")
skills = [s.strip().lower() for s in skills_input.split(",")]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
window_size = 3
cv_results = []

for pdf_filename in pdf_filenames:
    print(f"\nAnalyzing: {pdf_filename}")

    pdf_text = extract_text_from_pdf(pdf_filename)
    cleaned_text = preprocess(pdf_text)

    X = vectorizer.fit_transform([use_case_clean, cleaned_text])
    similarity_score = cosine_similarity(X[0:1], X[1:2])[0][0] * 100

    lines = pdf_text.split('\n')
    matched_snippets = []

    for skill in skills:
        found = False
        for i, line in enumerate(lines):
            if skill in line.lower():
                found = True
                start = max(i - window_size, 0)
                end = min(i + window_size + 3, len(lines))
                snippet = lines[start:end]
                snippet_block = "\n".join(snippet).strip()
                matched_snippets.append((skill, snippet_block))
        if not found:
            matched_snippets.append((skill, "Not found in this CV"))

    print(f"\nSkill Matching for {pdf_filename}:")
    for skill, snippet in matched_snippets:
        print(f"\n{skill.upper()}:\n{snippet}")

    matched_skill_names = sorted(set([s for s, snip in matched_snippets if "Not found" not in snip]))
    base_name = pdf_filename.replace('.pdf', '')

    with open(f"{base_name}_summary.txt", "w") as f:
        f.write(f"Use-case: {use_case}\n")
        f.write(f"Usefulness Score: {similarity_score:.2f}%\n")

    with open(f"{base_name}_skills.txt", "w") as f:
        f.write(f"Skills Checked: {', '.join(skills)}\n\n")
        for skill, snip in matched_snippets:
            f.write(f"\n--- Skill: {skill.upper()} ---\n{snip}\n")

    cv_results.append({
        'CV Filename': pdf_filename,
        'Score (%)': round(similarity_score, 2),
        'Skills Matched': ', '.join(matched_skill_names)
    })

df = pd.DataFrame(cv_results)
df_sorted = df.sort_values(by='Score (%)', ascending=False)

df_sorted.to_csv("CV_Analysis_Report.csv", index=False)
df_sorted.to_excel("CV_Analysis_Report.xlsx", index=False)

print("\nFinal Ranked Report:")
print(df_sorted)

print("\nFiles Saved:")
print("CV_Analysis_Report.csv")
print("CV_Analysis_Report.xlsx")
print("Individual *_summary.txt and *_skills.txt files")
