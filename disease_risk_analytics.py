import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🦠 DISEASE RISK & SEVERITY ANALYSIS")
print("=" * 40)

# Load your actual data
df = pd.read_csv("data/symptoms_disease_data.csv")

plt.figure(figsize=(16, 12))

# 1. DISEASE SEVERITY ANALYSIS (Based on your actual symptoms)
plt.subplot(2, 2, 1)

# Calculate average symptom severity for each disease
severity_metrics = df.groupby('disease').agg({
    'fever_severity': 'mean',
    'headache_severity': 'mean', 
    'cough_severity': 'mean',
    'body_pain': 'mean',
    'breathing_difficulty': 'mean'
}).mean(axis=1).sort_values(ascending=False)

# Create risk score (combination of multiple severe symptoms)
risk_factors = ['fever_severity', 'breathing_difficulty', 'body_pain', 'fever_duration']
df['risk_score'] = df[risk_factors].mean(axis=1)
disease_risk = df.groupby('disease')['risk_score'].mean().sort_values(ascending=False)

colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(disease_risk)))
bars = plt.bar(range(len(disease_risk)), disease_risk.values, color=colors)
plt.title('Disease Risk Level Analysis', fontweight='bold')
plt.xticks(range(len(disease_risk)), disease_risk.index, rotation=45)
plt.ylabel('Risk Score (Higher = More Severe)')

# Add risk labels
for i, (disease, score) in enumerate(disease_risk.items()):
    risk_level = "HIGH" if score > 2.0 else "MEDIUM" if score > 1.5 else "LOW"
    color = "red" if risk_level == "HIGH" else "orange" if risk_level == "MEDIUM" else "green"
    plt.text(i, score + 0.05, risk_level, ha='center', va='bottom', 
             fontweight='bold', color=color, fontsize=9)

# 2. SYMPTOM PATTERNS BY DISEASE
plt.subplot(2, 2, 2)

# Top symptoms for each disease
symptom_patterns = df.groupby('disease')[['fever_duration', 'cough_duration', 'body_pain']].mean()

sns.heatmap(symptom_patterns, annot=True, cmap='YlOrRd', fmt='.1f', 
            cbar_kws={'label': 'Symptom Intensity'})
plt.title('Key Symptom Patterns by Disease', fontweight='bold')
plt.ylabel('Disease')
plt.xlabel('Key Symptoms')

# 3. DISEASE FREQUENCY & URGENCY
plt.subplot(2, 2, 3)

disease_stats = df.groupby('disease').agg({
    'fever_duration': 'mean',
    'breathing_difficulty': 'mean',
    'age': 'count'  # Number of patients
}).rename(columns={'age': 'patient_count'})

# Create urgency score (combination of duration and severity)
disease_stats['urgency_score'] = (disease_stats['fever_duration'] * 0.4 + 
                                 disease_stats['breathing_difficulty'] * 0.6)

# Plot urgency vs frequency
plt.scatter(disease_stats['patient_count'], disease_stats['urgency_score'], 
           s=disease_stats['fever_duration']*50, alpha=0.7, 
           c=disease_stats['urgency_score'], cmap='Reds')

# Add labels
for disease, row in disease_stats.iterrows():
    plt.annotate(disease, (row['patient_count'], row['urgency_score']),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.xlabel('Number of Patients (Frequency)')
plt.ylabel('Urgency Score')
plt.title('Disease Frequency vs Urgency', fontweight='bold')
plt.colorbar(label='Urgency Level')

# 4. PREVENTION & TREATMENT INSIGHTS
plt.subplot(2, 2, 4)

# Analyze which diseases need immediate attention
prevention_data = []
for disease in df['disease'].unique():
    disease_data = df[df['disease'] == disease]
    avg_fever = disease_data['fever_duration'].mean()
    avg_breathing = disease_data['breathing_difficulty'].mean()
    patient_count = len(disease_data)
    
    if avg_breathing > 1.5:
        action = "IMMEDIATE CARE"
        color = "red"
    elif avg_fever > 4:
        action = "URGENT CONSULT"
        color = "orange"
    else:
        action = "MONITOR"
        color = "green"
    
    prevention_data.append({
        'Disease': disease,
        'Action': action,
        'Color': color,
        'Patients': patient_count
    })

prevention_df = pd.DataFrame(prevention_data)
colors_list = [prevention_df[prevention_df['Disease'] == d]['Color'].iloc[0] 
               for d in prevention_df['Disease']]

bars = plt.barh(prevention_df['Disease'], prevention_df['Patients'], color=colors_list)
plt.xlabel('Number of Patients')
plt.title('Recommended Medical Action by Disease', fontweight='bold')

# Add action labels
for i, (bar, row) in enumerate(zip(bars, prevention_df.iterrows())):
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
             row[1]['Action'], va='center', fontweight='bold', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=row[1]['Color'], alpha=0.3))

plt.tight_layout()
plt.savefig('disease_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Disease risk analysis completed!")
print("📊 Shows real risk levels based on your symptom data")
print("🎯 Provides actionable medical insights")