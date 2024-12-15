# Cancer Prediction Using Decision Tree Classifier

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

cancer_data = load_breast_cancer()

data = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
data['target'] = cancer_data.target

print("Dataset Preview:")
print(data.head())

# Split data into features 
X = data[cancer_data.feature_names]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


