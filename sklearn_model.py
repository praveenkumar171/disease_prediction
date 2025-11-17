import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("🤖 CREATING DISEASE PREDICTION MODEL WITH SCIKIT-LEARN...")

# 1. Load data
df = pd.read_csv("data/symptoms_disease_data.csv")
print(f"✅ Data loaded! {len(df)} patient records")

# 2. Show data info
print(f"📊 Diseases in dataset: {df['disease'].unique().tolist()}")
print(f"🔢 Features: {df.shape[1]-1} symptoms")

# 3. Prepare data
X = df.drop('disease', axis=1)  # All symptoms
y = df['disease']               # Disease labels

# Encode disease names to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("🔧 Training Random Forest model...")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.1%}")

# 7. Save everything
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sklearn_disease_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Model saved to 'models/sklearn_disease_model.pkl'")
print("✅ Label encoder saved to 'models/label_encoder.pkl'")

# 8. Show feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 TOP 5 MOST IMPORTANT SYMPTOMS:")
print(feature_importance.head(5))

# 9. Test prediction
sample_patient = X.iloc[0:1]  # First patient
prediction = model.predict(sample_patient)
predicted_disease = le.inverse_transform(prediction)

print(f"\n🧪 SAMPLE PREDICTION TEST:")
print(f"   Input: {X.iloc[0].to_dict()}")
print(f"   Actual: {df.iloc[0]['disease']}")
print(f"   Predicted: {predicted_disease[0]}")

print("\n🎉 DISEASE PREDICTION MODEL CREATED SUCCESSFULLY!")
print("💡 You can now use this model to predict diseases from symptoms!")