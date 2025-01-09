import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset and store it in SQLite database
df = pd.read_csv("278k_song_labelled.csv")

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("music_moods.db")

# Save the dataframe to SQLite database (create a table 'songs')
df.to_sql("songs", conn, if_exists="replace", index=False)

conn.close()

# Step 2: Query the data from the SQLite database with label mapping
conn = sqlite3.connect("music_moods.db")
query = """
    SELECT *,
           CASE 
               WHEN labels = 0 THEN 'sad'
               WHEN labels = 1 THEN 'happy'
               WHEN labels = 2 THEN 'energetic'
               WHEN labels = 3 THEN 'calm'
           END AS mood
    FROM songs
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Step 3: Normalize numerical features
features_to_normalize = [
    "duration (ms)", "danceability", "energy", "loudness", "speechiness", 
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "spec_rate"
]
scaler = StandardScaler()
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Step 4: Define the feature matrix (X) and target variable (y)
X = df[features_to_normalize]  # Features
y = df["mood"]  # Target variable

# Step 5: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Extract feature importance from the Random Forest model
feature_importance = model.feature_importances_

# Create a DataFrame for better readability
features = X.columns  # Feature names
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Save to a CSV file for PowerBI
importance_df.to_csv("feature_importance.csv", index=False)

# Generate the confusion matrix
moods = ["calm", "energetic", "happy", "sad"]
cm = confusion_matrix(y_test, y_pred, labels=moods)

# Convert to DataFrame for clarity
cm_df = pd.DataFrame(cm, index=moods, columns=moods)

# Plot the confusion matrix
cm_df.to_csv("confusion_matrix.csv")

# Parse the classification report into a dictionary
report_dict = classification_report(y_test, y_pred, target_names=["calm", "energetic", "happy", "sad"], output_dict=True)

# Convert the dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save the DataFrame as a CSV file
report_df.to_csv("classification_report.csv", index=True)

