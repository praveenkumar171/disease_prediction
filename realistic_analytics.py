import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

print("📊 REALISTIC ANALYTICS DASHBOARD")
print("=" * 40)

# Load data and model
df = pd.read_csv("data/symptoms_disease_data.csv")
model = joblib.load("models/sklearn_disease_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Prepare data
X = df.drop('disease', axis=1)
y = le.transform(df['disease'])

# Cross-validation for realistic accuracy
cv_scores = cross_val_score(model, X, y, cv=5)
real_accuracy = cv_scores.mean()

print(f"🎯 Realistic Accuracy (Cross-Validation): {real_accuracy:.1%}")

# Create realistic visualizations
plt.figure(figsize=(16, 10))

# 1. REAL Disease Distribution
plt.subplot(2, 3, 1)
disease_counts = df['disease'].value_counts()
plt.bar(disease_counts.index, disease_counts.values, color='lightblue', alpha=0.7)
plt.title('Disease Distribution (Imbalanced Data)')
plt.xticks(rotation=45)
plt.ylabel('Number of Patients')

# Add data labels
for i, count in enumerate(disease_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

# 2. REAL Symptom Importance (Correct)
plt.subplot(2, 3, 2)
feature_importance = pd.DataFrame({
    'Symptom': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Symptom'], feature_importance['Importance'], color='lightgreen')
plt.title('Top Predictive Symptoms ✓')
plt.xlabel('Importance Score')

# 3. REAL Cross-Validation Results
plt.subplot(2, 3, 3)
plt.bar(range(len(cv_scores)), cv_scores, color='orange', alpha=0.7)
plt.axhline(y=real_accuracy, color='red', linestyle='--', label=f'Average: {real_accuracy:.1%}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# Add accuracy labels
for i, score in enumerate(cv_scores):
    plt.text(i, score + 0.01, f'{score:.1%}', ha='center', va='bottom')

# 4. REAL Data Limitations
plt.subplot(2, 3, 4)
patients_per_disease = df['disease'].value_counts().values
plt.pie(patients_per_disease, labels=df['disease'].value_counts().index, autopct='%1.1f%%')
plt.title('Data Distribution - Limited Samples')

# 5. REAL Symptom Patterns Heatmap
plt.subplot(2, 3, 5)
symptom_patterns = df.groupby('disease').mean()[['fever_duration', 'cough_duration', 'body_pain']]
sns.heatmap(symptom_patterns, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Average Symptom Patterns by Disease')

# 6. REAL Model Performance Summary
plt.subplot(2, 3, 6)
performance_data = {
    'Metric': ['Training Accuracy', 'CV Accuracy', 'Diseases', 'Patients', 'Symptoms'],
    'Value': ['100%', f'{real_accuracy:.1%}', '11', '55', '17']
}
performance_df = pd.DataFrame(performance_data)
plt.table(cellText=performance_df.values,
          colLabels=performance_df.columns,
          cellLoc='center',
          loc='center',
          bbox=[0, 0, 1, 1])
plt.axis('off')
plt.title('Model Performance Summary')

plt.tight_layout()
plt.savefig('realistic_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🔍 REALISTIC INSIGHTS:")
print(f"• Cross-Validation Accuracy: {real_accuracy:.1%}")
print(f"• Data Limitation: Only {df['disease'].value_counts().min()} patients for some diseases")
print(f"• Most Important Symptom: {feature_importance.iloc[-1]['Symptom']}")
print(f"• Recommendation: Need more diverse medical data for production use")