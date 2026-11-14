"""
HealthAI - Machine Learning Model for Disease Risk Prediction
Trains and manages prediction models for diabetes and heart disease
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

class HealthAIModel:
    def __init__(self):
        self.diabetes_model = None
        self.heart_model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath):
        """Load patient data from CSV"""
        df = pd.read_csv(filepath)
        return df

    def prepare_data(self, df, target_column):
        """Prepare data for training"""
        # Remove rows with missing values
        df = df.dropna()

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train, X_test, y_test, model_type='diabetes'):
        """Train Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n{model_type.upper()} Model Performance:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        if model_type == 'diabetes':
            self.diabetes_model = model
        else:
            self.heart_model = model

        return model

    def predict_diabetes_risk(self, patient_data):
        """Predict diabetes risk for a patient"""
        if self.diabetes_model is None:
            return {"error": "Model not trained"}

        # Scale the input
        patient_scaled = self.scaler.transform([patient_data])

        # Get prediction and probability
        prediction = self.diabetes_model.predict(patient_scaled)[0]
        probability = self.diabetes_model.predict_proba(patient_scaled)[0]

        risk_score = probability[1] * 100  # Probability of disease
        risk_level = self._get_risk_level(risk_score)

        return {
            "disease": "Diabetes",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "probability": round(probability[1], 4),
            "recommendation": self._get_recommendation(risk_level)
        }

    def predict_heart_risk(self, patient_data):
        """Predict heart disease risk for a patient"""
        if self.heart_model is None:
            return {"error": "Model not trained"}

        # Scale the input
        patient_scaled = self.scaler.transform([patient_data])

        # Get prediction and probability
        prediction = self.heart_model.predict(patient_scaled)[0]
        probability = self.heart_model.predict_proba(patient_scaled)[0]

        risk_score = probability[1] * 100
        risk_level = self._get_risk_level(risk_score)

        return {
            "disease": "Heart Disease",
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "probability": round(probability[1], 4),
            "recommendation": self._get_recommendation(risk_level)
        }

    def _get_risk_level(self, score):
        """Categorize risk level based on score"""
        if score < 30:
            return "LOW"
        elif score < 70:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_recommendation(self, risk_level):
        """Get health recommendation based on risk level"""
        recommendations = {
            "LOW": "Continue healthy lifestyle. Regular check-ups recommended.",
            "MEDIUM": "Monitor health closely. Consult with healthcare provider.",
            "HIGH": "Immediate medical consultation recommended. Lifestyle changes needed."
        }
        return recommendations.get(risk_level, "Consult healthcare provider")

    def save_model(self, filename):
        """Save trained model to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'diabetes_model': self.diabetes_model,
                'heart_model': self.heart_model,
                'scaler': self.scaler,
                'features': self.feature_names
            }, f)

    def load_model(self, filename):
        """Load trained model from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.diabetes_model = data['diabetes_model']
            self.heart_model = data['heart_model']
            self.scaler = data['scaler']
            self.feature_names = data['features']


def train_models():
    """Main function to train all models"""
    model = HealthAIModel()

    # Load data
    df = model.load_data('sample_data.csv')
    print(f"Loaded {len(df)} patient records")

    # Train diabetes model
    print("\n" + "="*50)
    print("Training Diabetes Risk Prediction Model...")
    print("="*50)
    X_train, X_test, y_train, y_test = model.prepare_data(df, 'diabetes_risk')
    model.train_model(X_train, y_train, X_test, y_test, 'diabetes')

    # Train heart disease model
    print("\n" + "="*50)
    print("Training Heart Disease Risk Prediction Model...")
    print("="*50)
    X_train, X_test, y_train, y_test = model.prepare_data(df, 'heart_risk')
    model.train_model(X_train, y_train, X_test, y_test, 'heart')

    # Save models
    model.save_model('healthai_model.pkl')
    print("\n✓ Models trained and saved successfully!")

    return model


if __name__ == "__main__":
    train_models()
