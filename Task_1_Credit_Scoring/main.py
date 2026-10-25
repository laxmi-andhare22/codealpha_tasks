import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'Age': [25, 35, 45, 20, 35, 52, 23, 40, 60, 48],
    'Income': [50000, 60000, 80000, 20000, 120000, 90000, 30000, 70000, 100000, 65000],
    'Credit_History': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    'Approved': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['Age', 'Income', 'Credit_History']]
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")
print("CodeAlpha Task 1 - Credit Scoring")