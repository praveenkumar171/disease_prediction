from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and encoder
try:
    model = joblib.load("models/sklearn_disease_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    print("✅ Model loaded successfully!")
except:
    print("❌ Model not found! Please run sklearn_model.py first")
    exit()

# Disease information
disease_info = {
    'Common Cold': 'Mild viral infection with fever, cough, headache. Rest and hydration recommended.',
    'COVID-19': 'Respiratory illness with fever, cough, breathing difficulty. Medical consultation advised.',
    'Dengue': 'Mosquito-borne viral disease with high fever, severe body pain, rash. Immediate care needed.',
    'Malaria': 'Parasitic infection with intermittent fever, chills, sweating. Antimalarial treatment required.',
    'Typhoid': 'Bacterial infection with prolonged fever, headache, weakness. Antibiotic treatment needed.',
    'Pneumonia': 'Lung infection with cough, fever, breathing difficulty. Urgent medical attention required.',
    'Migraine': 'Neurological condition with severe headache, sensitivity to light/sound. Rest in dark room.',
    'Tuberculosis': 'Chronic bacterial infection with cough, weight loss, fever. Long-term treatment needed.',
    'Meningitis': 'Brain/spinal cord inflammation with severe headache, fever. EMERGENCY - seek immediate care.',
    'Chikungunya': 'Viral disease with fever, severe joint pain, rash. Pain management and rest.',
    'Influenza': 'Seasonal flu with fever, cough, body pain. Rest, fluids, and symptomatic treatment.'
}

def get_symptoms_summary(symptoms):
    """Create a human-readable summary of entered symptoms"""
    summary = []
    
    # Fever
    if symptoms['fever_duration'] > 0:
        fever_severity_map = {1: 'Mild', 2: 'Moderate', 3: 'High'}
        severity = fever_severity_map.get(symptoms['fever_severity'], 'No')
        summary.append(f"Fever: {severity} ({symptoms['fever_duration']} days)")
    else:
        summary.append("No Fever")
    
    # Headache
    if symptoms['headache_duration'] > 0:
        headache_severity_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        severity = headache_severity_map.get(symptoms['headache_severity'], 'No')
        summary.append(f"Headache: {severity} ({symptoms['headache_duration']} days)")
    else:
        summary.append("No Headache")
    
    # Cough
    if symptoms['cough_duration'] > 0:
        cough_type = "Dry" if symptoms['cough_type'] == 1 else "Productive"
        cough_severity_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        severity = cough_severity_map.get(symptoms['cough_severity'], 'No')
        summary.append(f"Cough: {cough_type} - {severity} ({symptoms['cough_duration']} days)")
    else:
        summary.append("No Cough")
    
    # Body Pain
    body_pain_map = {0: 'No', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    summary.append(f"Body Pain: {body_pain_map.get(symptoms['body_pain'], 'No')}")
    
    # Chills
    chills_map = {0: 'No', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    summary.append(f"Chills: {chills_map.get(symptoms['chills'], 'No')}")
    
    # Breathing Difficulty
    breathing_map = {0: 'No', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    summary.append(f"Breathing Difficulty: {breathing_map.get(symptoms['breathing_difficulty'], 'No')}")
    
    # Rash
    summary.append(f"Rash: {'Yes' if symptoms['rash'] == 1 else 'No'}")
    
    # Travel History
    summary.append(f"Recent Travel: {'Yes' if symptoms['travel_history'] == 1 else 'No'}")
    
    # Age
    summary.append(f"Age: {symptoms['age']} years")
    
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        symptoms = {
            'fever_duration': int(request.form.get('fever_duration', 0)),
            'fever_severity': int(request.form.get('fever_severity', 0)),
            'fever_onset': int(request.form.get('fever_onset', 0)),
            'fever_timing': int(request.form.get('fever_timing', 0)),
            'headache_duration': int(request.form.get('headache_duration', 0)),
            'headache_severity': int(request.form.get('headache_severity', 0)),
            'headache_onset': int(request.form.get('headache_onset', 0)),
            'headache_improves': int(request.form.get('headache_improves', 0)),
            'cough_duration': int(request.form.get('cough_duration', 0)),
            'cough_type': int(request.form.get('cough_type', 0)),
            'cough_severity': int(request.form.get('cough_severity', 0)),
            'age': int(request.form.get('age', 30)),
            'travel_history': int(request.form.get('travel_history', 0)),
            'body_pain': int(request.form.get('body_pain', 0)),
            'rash': int(request.form.get('rash', 0)),
            'chills': int(request.form.get('chills', 0)),
            'breathing_difficulty': int(request.form.get('breathing_difficulty', 0))
        }
        
        # Validate: Check if at least one symptom is provided
        total_symptoms = (
            symptoms['fever_duration'] +
            symptoms['fever_severity'] +
            symptoms['headache_duration'] +
            symptoms['headache_severity'] +
            symptoms['cough_duration'] +
            symptoms['cough_severity'] +
            symptoms['body_pain'] +
            symptoms['chills'] +
            symptoms['breathing_difficulty'] +
            symptoms['rash'] +
            symptoms['travel_history']
        )
        
        if total_symptoms == 0:
            error_msg = "⚠️ No symptoms provided! Please select at least one symptom (fever, headache, cough, body pain, chills, breathing difficulty, rash, or recent travel) before making a prediction."
            return render_template('error.html', error=error_msg)
        
        print(f"DEBUG: Form data received: {symptoms}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([symptoms])
        print(f"DEBUG: DataFrame shape: {input_df.shape}")
        print(f"DEBUG: DataFrame:\n{input_df}")
        
        # Make prediction
        prediction = model.predict(input_df)
        print(f"DEBUG: Prediction result: {prediction}")
        
        predicted_disease = le.inverse_transform(prediction)[0]
        print(f"DEBUG: Predicted disease: {predicted_disease}")
        
        # Get probabilities
        probabilities = model.predict_proba(input_df)[0]
        print(f"DEBUG: Probabilities shape: {probabilities.shape}")
        print(f"DEBUG: Probabilities: {probabilities}")
        
        disease_probabilities = {}
        
        for i, prob in enumerate(probabilities):
            disease_name = le.inverse_transform([i])[0]
            disease_probabilities[disease_name] = round(prob * 100, 2)
        
        print(f"DEBUG: Disease probabilities: {disease_probabilities}")
        
        # Sort by probability
        sorted_probabilities = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
        print(f"DEBUG: Sorted probabilities: {sorted_probabilities}")
        
        # Get all predictions for chart
        all_predictions = {disease: prob for disease, prob in sorted_probabilities}
        
        # Get top 3 predictions
        top_predictions = sorted_probabilities[:3]
        print(f"DEBUG: Top predictions: {top_predictions}")
        print(f"DEBUG: Top predictions length: {len(top_predictions)}")
        
        # Prepare summary of entered symptoms
        symptoms_summary = get_symptoms_summary(symptoms)
        
        return render_template('result.html', 
                             predicted_disease=predicted_disease,
                             top_predictions=top_predictions,
                             all_predictions=all_predictions,
                             disease_info=disease_info,
                             symptoms=symptoms,
                             symptoms_summary=symptoms_summary)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        return render_template('error.html', error=f"{str(e)}\n\n{error_details}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON responses"""
    try:
        data = request.get_json()
        symptoms = {
            'fever_duration': int(data['fever_duration']),
            'fever_severity': int(data['fever_severity']),
            # ... include all symptoms
        }
        
        input_df = pd.DataFrame([symptoms])
        prediction = model.predict(input_df)
        predicted_disease = le.inverse_transform(prediction)[0]
        
        probabilities = model.predict_proba(input_df)[0]
        disease_probabilities = {}
        
        for i, prob in enumerate(probabilities):
            disease_name = le.inverse_transform([i])[0]
            disease_probabilities[disease_name] = round(prob * 100, 2)
        
        return jsonify({
            'predicted_disease': predicted_disease,
            'probabilities': disease_probabilities,
            'description': disease_info.get(predicted_disease, 'No description available')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)