"""
HealthAI - Flask Web Application
Simple interface for disease risk prediction
"""

from flask import Flask, render_template, request, jsonify
from model import HealthAIModel
import os
import json

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load trained model
model = HealthAIModel()
if os.path.exists('healthai_model.pkl'):
    model.load_model('healthai_model.pkl')
else:
    print("Warning: Trained model not found. Please run model.py first.")

# Feature ranges for validation
FEATURE_RANGES = {
    'age': (18, 100),
    'bmi': (15, 50),
    'blood_pressure_systolic': (80, 200),
    'blood_pressure_diastolic': (40, 120),
    'glucose': (70, 400),
    'cholesterol': (100, 400),
    'heart_rate': (40, 150),
    'physical_activity_hours': (0, 24),
}


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - handles both GET and POST"""
    if request.method == 'GET':
        return render_template('predict.html')

    # Handle POST request (form submission)
    try:
        data = request.json

        # Validate input
        if not all(key in data for key in ['age', 'bmi', 'glucose']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert to float
        patient_data = {
            'age': float(data.get('age', 0)),
            'bmi': float(data.get('bmi', 0)),
            'blood_pressure_systolic': float(data.get('bp_systolic', 120)),
            'blood_pressure_diastolic': float(data.get('bp_diastolic', 80)),
            'glucose': float(data.get('glucose', 100)),
            'cholesterol': float(data.get('cholesterol', 200)),
            'heart_rate': float(data.get('heart_rate', 70)),
            'physical_activity_hours': float(data.get('activity_hours', 5)),
        }

        # Create feature vector in correct order
        feature_vector = [
            patient_data['age'],
            patient_data['bmi'],
            patient_data['blood_pressure_systolic'],
            patient_data['blood_pressure_diastolic'],
            patient_data['glucose'],
            patient_data['cholesterol'],
            patient_data['heart_rate'],
            patient_data['physical_activity_hours'],
        ]

        # Get predictions
        diabetes_result = model.predict_diabetes_risk(feature_vector)
        heart_result = model.predict_heart_risk(feature_vector)

        return jsonify({
            'success': True,
            'patient_data': patient_data,
            'diabetes': diabetes_result,
            'heart_disease': heart_result,
            'overall_risk': max(diabetes_result['risk_score'], heart_result['risk_score'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sample-patients', methods=['GET'])
def get_sample_patients():
    """Get sample patient data for testing"""
    samples = [
        {
            'name': 'Low Risk Patient',
            'age': 25,
            'bmi': 23,
            'bp_systolic': 120,
            'bp_diastolic': 80,
            'glucose': 95,
            'cholesterol': 180,
            'heart_rate': 65,
            'activity_hours': 7
        },
        {
            'name': 'Medium Risk Patient',
            'age': 45,
            'bmi': 28,
            'bp_systolic': 140,
            'bp_diastolic': 90,
            'glucose': 125,
            'cholesterol': 220,
            'heart_rate': 78,
            'activity_hours': 3
        },
        {
            'name': 'High Risk Patient',
            'age': 65,
            'bmi': 32,
            'bp_systolic': 160,
            'bp_diastolic': 100,
            'glucose': 180,
            'cholesterol': 280,
            'heart_rate': 85,
            'activity_hours': 1
        }
    ]
    return jsonify(samples)


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get app information"""
    return jsonify({
        'name': 'HealthAI',
        'version': '1.0.0',
        'track': 'AI + Health',
        'hackathon': 'Veritas AI International High School Hackathon',
        'features': [
            'Diabetes Risk Prediction',
            'Heart Disease Risk Prediction',
            'Patient Risk Stratification',
            'Real-time Health Assessment'
        ],
        'model_info': {
            'algorithm': 'Random Forest Classifier',
            'accuracy': '~85%',
            'training_samples': '5000+',
            'features': 8
        }
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════╗
    ║         HealthAI v1.0 Running          ║
    ║  Disease Risk Prediction System        ║
    ╚════════════════════════════════════════╝

    🌐 Open browser: http://localhost:5000
    📊 Dashboard: http://localhost:5000/
    🔍 Predict: http://localhost:5000/predict
    """)
    app.run(debug=True, port=5000)
