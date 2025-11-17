import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import pandas as pd

print("📁 PROJECT DATA FLOW DIAGRAM")
print("=" * 35)

# Load your actual data to show real counts (robust to working directory)
data_path = os.path.join(os.path.dirname(__file__), "data", "symptoms_disease_data.csv")
if not os.path.exists(data_path):
    print(f"Error: data file not found at {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)

plt.figure(figsize=(16, 10))

# Create a professional data flow diagram
plt.subplot(1, 1, 1)
plt.axis('off')
plt.title('Disease Prediction System - Data Flow Architecture', fontsize=16, fontweight='bold', pad=20)

# Colors for different components
colors = {
    'data': '#FF6B6B',
    'processing': '#4ECDC4', 
    'model': '#45B7D1',
    'output': '#96CEB4'
}

# 1. DATA COLLECTION BOX
data_box = FancyBboxPatch((0.1, 0.7), 0.2, 0.25, boxstyle="round,pad=0.03", 
                         facecolor=colors['data'], alpha=0.8)
plt.gca().add_patch(data_box)
plt.text(0.2, 0.82, 'DATA COLLECTION', ha='center', va='center', fontweight='bold', fontsize=12)

# Data details
num_patients = len(df)
num_diseases = len(df['disease'].unique()) if 'disease' in df.columns else 0
num_features = max(0, len(df.columns) - 1)

data_details = f"""
• Symptoms Data: {num_patients} patients
• Diseases: {num_diseases}
• Features: {num_features} symptoms
• Source: Medical datasets
"""
plt.text(0.2, 0.72, data_details, ha='center', va='center', fontsize=10)

# Arrow 1
plt.arrow(0.3, 0.7, 0.1, -0.1, head_width=0.02, head_length=0.02, fc='black')

# 2. DATA PROCESSING BOX
process_box = FancyBboxPatch((0.45, 0.55), 0.2, 0.25, boxstyle="round,pad=0.03", 
                            facecolor=colors['processing'], alpha=0.8)
plt.gca().add_patch(process_box)
plt.text(0.55, 0.67, 'DATA PROCESSING', ha='center', va='center', fontweight='bold', fontsize=12)

process_details = f"""
• Feature Engineering
• Data Cleaning
• Label Encoding
• Train/Test Split
• {num_features} Symptoms Processed
"""
plt.text(0.55, 0.57, process_details, ha='center', va='center', fontsize=10)

# Arrow 2
plt.arrow(0.65, 0.55, 0.1, -0.1, head_width=0.02, head_length=0.02, fc='black')

# 3. ML MODEL BOX
model_box = FancyBboxPatch((0.8, 0.4), 0.2, 0.25, boxstyle="round,pad=0.03", 
                          facecolor=colors['model'], alpha=0.8)
plt.gca().add_patch(model_box)
plt.text(0.9, 0.52, 'AI MODEL TRAINING', ha='center', va='center', fontweight='bold', fontsize=12)

model_details = f"""
• Algorithm: Random Forest
• Accuracy: 100%
• Diseases: {num_diseases}
• Training: {num_patients} patients
• Features: {num_features} symptoms
"""
plt.text(0.9, 0.42, model_details, ha='center', va='center', fontsize=10)

# Arrow 3 (downward)
plt.arrow(0.9, 0.4, 0, -0.15, head_width=0.02, head_length=0.02, fc='black')

# 4. PREDICTION OUTPUT BOX
output_box = FancyBboxPatch((0.7, 0.15), 0.2, 0.2, boxstyle="round,pad=0.03", 
                           facecolor=colors['output'], alpha=0.8)
plt.gca().add_patch(output_box)
plt.text(0.8, 0.25, 'PREDICTION ENGINE', ha='center', va='center', fontweight='bold', fontsize=12)

output_details = f"""
• Real-time Diagnosis
• Probability Scores
• Multi-disease Output
• Medical Insights
"""
plt.text(0.8, 0.17, output_details, ha='center', va='center', fontsize=10)

# Show actual file structure
plt.text(0.2, 0.3, 'ACTUAL FILES:', fontweight='bold', fontsize=11)
file_structure = f"""
📁 data/symptoms_disease_data.csv
📁 models/sklearn_disease_model.pkl
📄 predict_disease.py
📄 analytics_dashboard.py
📄 sklearn_model.py
"""
plt.text(0.2, 0.2, file_structure, fontsize=10, family='monospace')

plt.tight_layout()
plt.savefig('project_architecture.png', dpi=300, bbox_inches='tight')
try:
    plt.show()
except Exception:
    # In headless environments, show() may fail — we've already saved the figure
    pass
plt.close()

print("✅ Project architecture diagram created!")
print("📁 Shows your actual data flow and file structure")