# Task 10: KNN – Handwritten Digit Classification

## 📌 Overview
This project implements a **K-Nearest Neighbors (KNN)** classifier to recognize handwritten digits using the **Scikit-learn Digits dataset**.  
It demonstrates how distance-based models work, the importance of feature scaling, and how to tune hyperparameters (`K`) for best performance.

---

## 🛠 Tools & Technologies
- Python  
- Scikit-learn  
- Matplotlib  

---

## 📊 Dataset
**Sklearn Digits Dataset**

```python
from sklearn.datasets import load_digits
digits = load_digits()
Images: 8×8 grayscale digits

Features: 64 pixel values

Classes: 0–9

📂 Project Structure
task-10-knn-digit-classification/
│
├── notebooks/
│   └── Task10_KNN_Digits.ipynb
│
├── visuals/
│   ├── accuracy_vs_k.png
│   ├── confusion_matrix.png
│
├── README.md
└── requirements.txt
🔹 Step 1: Load Dataset & Inspect
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

print("X shape:", X.shape)
print("y shape:", y.shape)
🔹 Step 2: Visualize Sample Digits
import matplotlib.pyplot as plt

plt.figure(figsize=(8,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {digits.target[i]}")
    plt.axis("off")
plt.show()
🔹 Step 3: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
🔹 Step 4: Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
🔹 Step 5: Train KNN (K=3)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy (K=3):", accuracy_score(y_test, y_pred))
🔹 Step 6: Try Multiple K Values
import numpy as np

k_values = [3,5,7,9]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    accuracies.append(acc)

print(dict(zip(k_values, accuracies)))
🔹 Step 7: Accuracy vs K Plot
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()
🔹 Step 8: Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
🔹 Step 9: Display Predictions
plt.figure(figsize=(8,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis("off")
plt.show()
📈 Sample Results (Typical)
K	Accuracy
3	0.98
5	0.99
7	0.98
9	0.97
Best K → 5
