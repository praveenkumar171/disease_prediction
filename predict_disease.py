import pandas as pd
import joblib
import os

print("🔮 DISEASE PREDICTION INTERFACE")
print("=" * 40)

# Load the saved model and encoder
model = joblib.load("models/sklearn_disease_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Disease descriptions
disease_info = {
    'Common Cold': 'Mild viral infection with fever, cough, headache',
    'COVID-19': 'Respiratory illness with fever, cough, breathing difficulty',
    'Dengue': 'Mosquito-borne with high fever, severe body pain, rash',
    'Malaria': 'Parasitic infection with intermittent fever, chills',
    'Typhoid': 'Bacterial infection with prolonged fever, headache',
    'Pneumonia': 'Lung infection with cough, fever, breathing difficulty',
    'Migraine': 'Severe headache without fever',
    'Tuberculosis': 'Chronic cough, weight loss, prolonged fever',
    'Meningitis': 'Severe headache, fever, neck stiffness',
    'Chikungunya': 'Fever with severe joint pain, rash',
    'Influenza': 'Seasonal flu with fever, cough, body pain'
}

def predict_disease():
    print("\n📝 ENTER PATIENT SYMPTOMS:")
    
    # Get symptoms from user
    symptoms = {
        'fever_duration': int(input("Fever duration (days): ")),
        'fever_severity': int(input("Fever severity (0=None, 1=Mild, 2=Moderate, 3=High): ")),
        'fever_onset': int(input("Fever onset (1=Sudden, 2=Gradual): ")),
        'fever_timing': int(input("Fever timing (1=Morning, 2=Evening, 3=Throughout day): ")),
        'headache_duration': int(input("Headache duration (days): ")),
        'headache_severity': int(input("Headache severity (0-3): ")),
        'headache_onset': int(input("Headache onset (1=Sudden, 2=Gradual): ")),
        'headache_improves': int(input("Headache improves with (1=Rest, 2=Medication, 3=Nothing): ")),
        'cough_duration': int(input("Cough duration (days): ")),
        'cough_type': int(input("Cough type (1=Dry, 2=Productive): ")),
        'cough_severity': int(input("Cough severity (0-3): ")),
        'age': int(input("Patient age: ")),
        'travel_history': int(input("Travel history (0=No, 1=Yes): ")),
        'body_pain': int(input("Body pain severity (0-3): ")),
        'rash': int(input("Rash (0=No, 1=Yes): ")),
        'chills': int(input("Chills severity (0-3): ")),
        'breathing_difficulty': int(input("Breathing difficulty (0-3): "))
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([symptoms])
    
    # Make prediction
    prediction = model.predict(input_df)
    predicted_disease = le.inverse_transform(prediction)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(input_df)[0]
    disease_probabilities = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
    
    # Show results
    print(f"\n🔍 PREDICTION RESULTS:")
    print(f"🎯 Most Likely: {predicted_disease}")
    print(f"📖 Description: {disease_info[predicted_disease]}")
    
    print(f"\n📊 ALL POSSIBILITIES:")
    for disease, prob in sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True):
        if prob > 0.01:  # Show only probabilities > 1%
            print(f"   {disease}: {prob:.1%}")

# Test the prediction
if __name__ == "__main__":
    predict_disease()