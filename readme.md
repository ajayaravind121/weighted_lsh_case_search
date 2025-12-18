# Medical Similar Case Search Engine (Weighted LSH)
Course

# CS 425 – Data Mining
Instructor: Prof. Zeyun Yu

# Team Members

- Sushma Ravichandran – Project design, system architecture, data preprocessing, core similarity algorithm, UI integration

- Ajay Aravind Prakash – Algorithm validation, similarity testing, LSH exploration, debugging and performance analysis

- Kavindu Ransika Wimalaratne – Dataset discovery, dataset exploration, UI testing, documentation support

# How to Run Locally

- Clone the Repository
https://github.com/ajayaravind121/weighted_lsh_case_search.git

- Install Dependencies
pip install -r requirements.txt

- Run the Application
streamlit run src/app.py

- Access the App

Your browser will automatically open at: http://localhost:8501

🌐 Deployed Application

The application is deployed on Streamlit Community Cloud and can be accessed here:

🔗 Live Demo: https://weightedlshsimilarcase.streamlit.app/


#  Project Overview

Healthcare systems store large volumes of patient information as unstructured clinical notes.
Keyword-based search fails to capture true clinical similarity because similar cases may use different terminology.

This project implements a content-based medical case search engine that:

- Accepts a clinical note as input

- Analyzes its textual content

- Retrieves the Top-K most similar past medical cases

- Uses weighted text similarity and Locality Sensitive Hashing (LSH) for scalability

- Provides results through an interactive web interface

#  Dataset

We use the MT Samples Medical Transcription Dataset, which contains real clinical documentation across multiple medical specialties.

- Dataset location: data/mtsamples.csv


# Fields used:

- Description

- Transcription

- Keywords

- Medical Specialty

These fields are combined to form a single searchable text representation for each medical case.

# Algorithms & Techniques Used (Step-by-Step)
1️⃣ Text Preparation

Relevant text fields (description, transcription, keywords) are combined.

Each medical case becomes a single document.

Purpose: Capture full clinical context.

2️⃣ TF-IDF Vectorization

Text documents are converted into numerical vectors using TF-IDF.

Important medical terms receive higher weights.

Common words receive lower weights.

Purpose:
Convert unstructured medical text into weighted numerical representations.

3️⃣ Weighted MinHash Signature Generation

TF-IDF vectors are compressed into fixed-length fingerprints.

Important terms influence the fingerprint more than common terms.

Purpose:
Reduce dimensionality while preserving similarity.

4️⃣ Locality Sensitive Hashing (LSH) with Banding

MinHash signatures are divided into bands.

Similar documents are grouped into the same buckets.

Only candidate cases from matching buckets are considered.

Purpose:
Improve scalability by avoiding comparison with all documents.

5️⃣ Final Similarity Computation

For shortlisted candidates, weighted similarity is computed using TF-IDF vectors.

Similarity scores reflect overlap of important medical terms.

Purpose:
Accurate ranking of similar cases.

6️⃣ Top-K Ranking

Cases are sorted by similarity score.

The system returns the Top-K most similar cases.

Purpose:
Present only the most relevant results to the user.


# Evaluation Note

This project is a similarity-based retrieval system, not a classification model.
Therefore, traditional metrics such as accuracy, precision, recall, F1-score, and ROC-AUC are not directly applicable without labeled relevance data.

Evaluation is based on:

- Quality of similarity ranking

- Clinical relevance of retrieved cases

- End-to-end system correctness

# Future Work

- Add image-to-text input for faster clinical note entry

- Implement adaptive LSH to automatically tune banding parameters

- Extend similarity search to medication or treatment recommendation tasks

- Introduce labeled relevance data to enable quantitative retrieval metrics (Precision@K, Recall@K)

# Conclusion

This project demonstrates a complete and correct application of data mining techniques to real-world medical text.
By moving beyond keyword search and integrating TF-IDF, Weighted MinHash, and LSH into an interactive system, the project validates the practicality of similarity-based medical case retrieval.