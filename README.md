# Classify_book_genre_202401100300209
# 📚 Book Genre Classification using Machine Learning

This project demonstrates how to classify book genres using metadata such as author popularity, book length, and the number of keywords. The goal is to predict genres like *mystery*, *fantasy*, *fiction*, and *non-fiction* using a supervised learning approach.

---

## 🔍 Overview

- **Input Data**: A dataset of 100 books with metadata features.
- **Model Used**: Random Forest Classifier (sklearn)
- **Visualization**: Confusion Matrix & Feature Importance (using Seaborn and Matplotlib)
- **Output**: Predicted genres with performance metrics

---

## 📁 Dataset Features

| Feature             | Description                                |
|---------------------|--------------------------------------------|
| `author_popularity` | Numerical score representing author's fame |
| `book_length`       | Number of pages in the book                |
| `num_keywords`      | Count of keywords linked to the book       |
| `genre`             | Target variable (genre label)              |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sagar-0401/Classify_book_genre_202401100300209/tree/main
cd book-genre-classification


“AI BASED CLASSIFY BOOK GENRES USE METADATA SUCH AS AUTHOR, LENGTH, AND KEYWORDS TO CLASSIFY BOOK GENRE.”
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("/content/book_genres.csv")

# Define features and target
X = df[['author_popularity', 'book_length', 'num_keywords']]
y = df['genre']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Print confusion matrix
print("=== Confusion Matrix ===")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.tight_layout()
plt.show()

# Plot feature importances
importances = clf.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
