import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("📊 LIVE ANALYTICS - REAL DATA")
print("=" * 35)

# Load your ACTUAL data
df = pd.read_csv("data/symptoms_disease_data.csv")
print(f"✅ Loaded {len(df)} patient records")

# Use your ACTUAL model
try:
    model = joblib.load("models/sklearn_disease_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    print("✅ Loaded your trained model")
except:
    print("🔄 Training new model for demo...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X = df.drop('disease', axis=1)
    y = le.fit_transform(df['disease'])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

# Create DYNAMIC charts based on ACTUAL predictions
plt.figure(figsize=(15, 10))

# 1. ACTUAL Disease Distribution from YOUR data
plt.subplot(2, 2, 1)
disease_counts = df['disease'].value_counts()
colors = plt.cm.viridis(np.linspace(0, 1, len(disease_counts)))
bars = plt.bar(range(len(disease_counts)), disease_counts.values, color=colors)
plt.title('Your Actual Disease Data')
plt.xticks(range(len(disease_counts)), disease_counts.index, rotation=45)
plt.ylabel('Number of Patients')

# Add actual numbers on bars
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             str(int(bar.get_height())), ha='center', va='bottom')

# 2. ACTUAL Symptom Importance from YOUR model
plt.subplot(2, 2, 2)
feature_importance = pd.DataFrame({
    'Symptom': df.drop('disease', axis=1).columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Symptom'], feature_importance['Importance'], 
         color='lightgreen', alpha=0.7)
plt.title('Your Model - Real Symptom Importance')
plt.xlabel('Importance Score')

# 3. ACTUAL Prediction Test - Use REAL predictions
plt.subplot(2, 2, 3)
# Take first 5 patients and show ACTUAL vs PREDICTED
test_patients = df.head(5)
X_test = test_patients.drop('disease', axis=1)
y_actual = test_patients['disease']
y_pred = le.inverse_transform(model.predict(X_test))

results = []
for i in range(len(test_patients)):
    results.append({
        'Patient': i+1,
        'Actual': y_actual.iloc[i],
        'Predicted': y_pred[i],
        'Match': y_actual.iloc[i] == y_pred[i]
    })

results_df = pd.DataFrame(results)
colors = ['lightgreen' if x else 'lightcoral' for x in results_df['Match']]
plt.bar(results_df['Patient'], [1]*len(results_df), color=colors, alpha=0.7)
plt.title('Real Prediction Test (First 5 Patients)')
plt.ylabel('Correct ✓ / Incorrect ✗')
plt.xticks(results_df['Patient'])

# Add labels
for i, row in results_df.iterrows():
    status = "✓" if row['Match'] else "✗"
    plt.text(row['Patient'], 0.5, status, ha='center', va='center', fontweight='bold')

# 4. ACTUAL Performance Metrics
plt.subplot(2, 2, 4)
# Calculate real accuracy on your data
X_all = df.drop('disease', axis=1)
y_all = le.transform(df['disease'])
y_all_pred = model.predict(X_all)
accuracy = (y_all_pred == y_all).mean()

metrics_data = {
    'Metric': ['Total Patients', 'Diseases', 'Symptoms', 'Your Accuracy'],
    'Value': [len(df), len(df['disease'].unique()), len(df.columns)-1, f'{accuracy:.1%}']
}

plt.table(cellText=pd.DataFrame(metrics_data).values,
          colLabels=pd.DataFrame(metrics_data).columns,
          cellLoc='center',
          loc='center',
          bbox=[0.1, 0.1, 0.8, 0.8])
plt.axis('off')
plt.title('Your Project - Real Metrics')

plt.tight_layout()
plt.savefig('live_analytics.png', dpi=300, bbox_inches='tight')  # FIXED: ddi -> dpi
plt.show()

print(f"\n🎯 YOUR ACTUAL MODEL PERFORMANCE:")
print(f"• Accuracy on your data: {accuracy:.1%}")
print(f"• Diseases predicted: {len(df['disease'].unique())}")
print(f"• Patients in dataset: {len(df)}")
print(f"• Most important symptom: {feature_importance.iloc[-1]['Symptom']}")