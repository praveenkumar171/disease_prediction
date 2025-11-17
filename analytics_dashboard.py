import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

print("📊 DISEASE PREDICTION ANALYTICS DASHBOARD")
print("=" * 50)

# Load data and model
df = pd.read_csv("data/symptoms_disease_data.csv")
model = joblib.load("models/sklearn_disease_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Prepare data
X = df.drop('disease', axis=1)
y = le.transform(df['disease'])

# Predictions
y_pred = model.predict(X)

# 1. Disease Distribution
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
df['disease'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Disease Distribution in Dataset')
plt.xticks(rotation=45)

# 2. Feature Importance
plt.subplot(2, 2, 2)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Symptom Importance')

# 3. Confusion Matrix
plt.subplot(2, 2, 3)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 4. Accuracy by Disease
plt.subplot(2, 2, 4)
accuracy_by_disease = []
for i, disease in enumerate(le.classes_):
    mask = y == i
    if mask.any():
        acc = (y_pred[mask] == i).mean()
        accuracy_by_disease.append((disease, acc))

diseases, accuracies = zip(*accuracy_by_disease)
plt.bar(diseases, accuracies, color='lightgreen')
plt.title('Accuracy by Disease')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('disease_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Analytics dashboard created!")
print("📈 Check 'disease_analytics.png' for visualizations")

# Show model performance
print(f"\n🎯 MODEL PERFORMANCE:")
print(f"Overall Accuracy: {(y_pred == y).mean():.1%}")
print(f"Diseases predicted: {len(le.classes_)}")
print(f"Features used: {X.shape[1]} symptoms")
