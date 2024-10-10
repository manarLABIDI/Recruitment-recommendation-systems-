import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import tempfile
import os
import time
import re
import spacy
from docx import Document
from pyresparser import ResumeParser
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_sm')


def extract_text_from_pdf(file):
    doc = Document()
    with open(file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            doc.add_paragraph(page.extract_text())

        # Extract text from all pages

    # Extract text from all pages
    doc.save("text.docx")
    data = ResumeParser('text.docx').get_extracted_data()

    # Extract specific fields
    skills = ' '.join(data['skills'])
    title = ' '.join(data['designation'])
    experience = str(data['total_experience'])

    return skills, title, experience


def main():

    st.set_page_config(page_title="Resume Matcher", page_icon=":clipboard:")
    st.title("Resume Matcher")
    # Make sure the image file extension is correct
    container = st.container()
    logo_image = "logo-Hydatis-sans-slogan.png"
    container.image(logo_image, width=200)

    st.markdown(
        f"""
         
        <style>
        .main {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .title {{
            text-align: center;
            font-size: 32px;
            margin-bottom: 20px;
        }}
        .subtitle {{
            text-align: center;
            font-size: 18px;
            margin-bottom: 20px;
        }}
        .upload-btn-wrapper {{
            position: relative;
            overflow: hidden;
            display: inline-block;
        }}
        .btn {{
            border: 2px solid gray;
            color: gray;
            background-color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }}
        .upload-btn-wrapper input[type=file] {{
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
        }}
        .container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
     .logo-img {{
        width: 100px;
        margin-right: 20px;
    }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    cleanedJ_data = pd.read_csv("cleanedJ_data.csv")
    cleanedR_data = pd.read_csv("updated_databaseResume.csv")

    cleanedR_data = cleanedR_data.dropna()
    cleanedJ_data = cleanedJ_data.dropna()

# Convertir toutes les colonnes en type str
    cleanedR_data = cleanedR_data.astype(str)
    cleanedJ_data = cleanedJ_data.astype(str)
    # Réduire la taille de df_reducedJ pour correspondre à la taille de cleanedR_data
    cleanedJ_data = cleanedJ_data.iloc[:, :cleanedR_data.shape[1]]
    cleanedJ_data = cleanedJ_data.iloc[:cleanedR_data.shape[0], :]

# Diviser les données en ensemble de train et de test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        cleanedR_data, cleanedJ_data, test_size=0.2, random_state=42)

# Créer une seule instance de CountVectorizer pour toutes les colonnes
    cv = CountVectorizer()

# Vectorisation des données de train
    count_train = cv.fit_transform(
        X_train['title'] + ' ' + X_train['required_skills'])
    tfidf_train = TfidfTransformer().fit_transform(count_train)
    tfidf_matrixR_train = hstack([tfidf_train])
# Vectorisation des données de test
    count_test = cv.transform(
        X_test['title'] + ' ' + X_test['required_skills'])
    tfidf_test = TfidfTransformer().fit_transform(count_test)
    tfidf_matrixJ = hstack([tfidf_test])
# Vectorisation des données d'offres d'emploi
    count_J = cv.transform(
        cleanedJ_data['title'] + ' ' + cleanedJ_data['required_skills'])
    tfidf_J = TfidfTransformer().fit_transform(count_J)
    tfidf_matrixJ = hstack([tfidf_J])

    @st.cache_data
    @st.cache_resource
    def match_jobs(cv_title, cv_skills, cv_experience, job_data, num_jobs=10):
        # Create a list of column names for the DataFrame
        columns = ["job_title", "job_skills",
                   "job_experience", "matching_score"]

    # Create a list to store the data for each row
        rows = []

    # Combine CV title, skills, and experience into a single string
        cv_text = cv_title + " " + cv_skills + " " + cv_experience

    # Create a TF-IDF vectorizer and fit it with CV and job data
        vectorizer = TfidfVectorizer()
        tfidf_matrixCV = vectorizer.fit_transform([cv_text])
        tfidf_matrixJ_skills = vectorizer.transform(
            job_data["required_skills"])
        tfidf_matrixJ_experience = vectorizer.transform(job_data["experience"])
        tfidf_matrixJ_title = vectorizer.transform(job_data["title"])

    # Calculate cosine similarity between CV and job data
        similarity_scores_skills = cosine_similarity(
            tfidf_matrixCV, tfidf_matrixJ_skills)[0]
        similarity_scores_experience = cosine_similarity(
            tfidf_matrixCV, tfidf_matrixJ_experience)[0]
        similarity_scores_title = cosine_similarity(
            tfidf_matrixCV, tfidf_matrixJ_title)[0]

    # Combine the similarity scores with weighted averages (e.g., 0.5 for skills, 0.3 for experience, and 0.2 for title)
        similarity_scores_combined = 0.5 * similarity_scores_skills + 0.7 * \
            similarity_scores_experience + 0.5 * similarity_scores_title

    # Sort the similarity scores in descending order and retrieve the corresponding indices
        top_matching_indices = np.argsort(similarity_scores_combined)[
            ::-1][:num_jobs]

    # Iterate through the top matching job indices
        for index in top_matching_indices:
            # Retrieve the information of the job offer
            job_title = job_data.iloc[index]["title"]
            job_skills = job_data.iloc[index]["required_skills"]
            job_experience = job_data.iloc[index]["experience"]
            matching_score = similarity_scores_combined[index]

        # Add a new row with the information to the list of rows
            rows.append(
                [job_title, job_skills, job_experience, matching_score])

    # Create a DataFrame with the collected rows and column names
        matching_df = pd.DataFrame(rows, columns=columns)

        return matching_df
    with st.container():
        st.markdown("<div class='main'>", unsafe_allow_html=True)

        st.markdown(
            "<div class='title'>Upload your resume (PDF format) below:</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"], key="resume")
        if uploaded_file:

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())

            skills, title, experience = extract_text_from_pdf(tmp_path)
            os.remove(tmp_path)

            st.markdown("<div class='subtitle'>Resume Text:</div>",
                        unsafe_allow_html=True)
            resume_text_placeholder = st.empty()
            resume_text_placeholder.write(
                'Title: '+title+'\n'+' Experience: '+experience+'\n'+' Skills: '+skills)

            if st.button("Match Jobs"):

                with st.spinner("Matching jobs..."):
                    time.sleep(2)

                    # Perform the matching using the loaded model
                    # You'll need to modify this code based on your model's input and output format

                    # Get the indices of the matched jobs
                    matched_jobs = match_jobs(
                        title, skills, experience, y_train)

                    # Display the matched job titles
                    st.markdown(
                        "<div class='subtitle'>Matched Jobs:</div>", unsafe_allow_html=True)
                    st.dataframe(matched_jobs)

        # Search section
        st.markdown("<div class='title'>Search for Job Titles:</div>",
                    unsafe_allow_html=True)
        search_query = st.text_input("", key="search_query")
        if st.button("Search"):
            # Show a spinner while performing the search
            with st.spinner("Searching jobs..."):
                time.sleep(2)  # Placeholder for actual search code

                st.markdown(
                    "<div class='subtitle'>Search Results:</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
