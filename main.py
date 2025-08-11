import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

b_cancer = load_breast_cancer()
X = b_cancer.data  
y = b_cancer.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=b_cancer.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

sample = [X_test[0]] 
predicted_class = model.predict(sample)
print(f"\nPredicted class for {sample} is: {b_cancer.target_names[predicted_class][0]}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=b_cancer.target_names,
            yticklabels=b_cancer.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
plt.show()



