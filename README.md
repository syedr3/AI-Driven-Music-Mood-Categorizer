# AI-Driven Music Mood Categorizer

An AI-driven system that predicts music mood categories using machine learning. This project leverages a Kaggle dataset of audio features, a Random Forest classifier, and an SQLite database for efficient data management. The results are visualized using PowerBI for insightful analysis.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)

## Features
- Predicts music moods: **calm**, **energetic**, **happy**, and **sad**.
- Utilizes a Random Forest classifier trained on audio features such as danceability, energy, loudness, and tempo.
- Data stored and managed using an SQLite database.
- Visualizes key insights and trends with PowerBI, including:
  - Mood distribution.
  - Feature importance analysis.
  - Classification results.

## Dataset
The dataset contains the following features:
- **Duration** (ms)
- **Danceability**
- **Energy**
- **Loudness**
- **Speechiness**
- **Acousticness**
- **Instrumentalness**
- **Liveness**
- **Valence**
- **Tempo**
- **Spectral Rate**
- **Labels**: `0` (Sad), `1` (Happy), `2` (Energetic), `3` (Calm)

## Project Workflow

1. **Data Preparation**:
    - Load the dataset into an SQLite database.
    - Retrieve and normalize relevant features.

2. **Model Training**:
    - Train a Random Forest Classifier on the processed dataset.

3. **Evaluation**:
    - Generate a classification report and confusion matrix.
    - Save these metrics for further analysis.

4. **Visualization**:
    - Create PowerBI reports to visualize insights like:
      - Mood distribution across the dataset.
      - Feature importance in predicting moods.
      - Confusion matrix showcasing model performance.

## Technologies Used
- **Python**: Data preprocessing, machine learning, and evaluation.
- **SQLite**: Database for efficient data management.
- **PowerBI**: Visualization of results and insights.
- **Pandas**: Data manipulation and analysis.
- **scikit-learn**: Model training and evaluation.
