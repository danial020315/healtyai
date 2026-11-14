"""
Data Processor - Generate sample patient data for training
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def generate_sample_data(num_samples=5000, filename='sample_data.csv'):
    """Generate synthetic patient dataset for training"""

    print(f"Generating {num_samples} synthetic patient records...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate base features
    age = np.random.randint(18, 85, num_samples)
    bmi = np.random.normal(25, 5, num_samples)  # Normal distribution around 25
    bmi = np.clip(bmi, 15, 50)  # Keep realistic range

    blood_pressure_systolic = np.random.normal(130, 15, num_samples)
    blood_pressure_systolic = np.clip(blood_pressure_systolic, 90, 200)

    blood_pressure_diastolic = np.random.normal(80, 10, num_samples)
    blood_pressure_diastolic = np.clip(blood_pressure_diastolic, 50, 120)

    glucose = np.random.normal(120, 30, num_samples)
    glucose = np.clip(glucose, 70, 400)

    cholesterol = np.random.normal(200, 40, num_samples)
    cholesterol = np.clip(cholesterol, 100, 400)

    heart_rate = np.random.normal(72, 15, num_samples)
    heart_rate = np.clip(heart_rate, 40, 150)

    physical_activity_hours = np.random.normal(4, 3, num_samples)
    physical_activity_hours = np.clip(physical_activity_hours, 0, 24)

    # Generate target variables based on risk factors
    # Diabetes risk: higher BMI, glucose, age increase risk
    diabetes_risk = (
        (age > 45) * 0.3 +
        (bmi > 28) * 0.4 +
        (glucose > 125) * 0.4 +
        (physical_activity_hours < 3) * 0.2
    ) > 0.6

    # Heart disease risk: higher BP, age, cholesterol, lower activity
    heart_risk = (
        (age > 50) * 0.3 +
        (blood_pressure_systolic > 140) * 0.4 +
        (cholesterol > 240) * 0.3 +
        (physical_activity_hours < 3) * 0.3
    ) > 0.6

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi.round(2),
        'blood_pressure_systolic': blood_pressure_systolic.round(0).astype(int),
        'blood_pressure_diastolic': blood_pressure_diastolic.round(0).astype(int),
        'glucose': glucose.round(0).astype(int),
        'cholesterol': cholesterol.round(0).astype(int),
        'heart_rate': heart_rate.round(0).astype(int),
        'physical_activity_hours': physical_activity_hours.round(1),
        'diabetes_risk': diabetes_risk.astype(int),
        'heart_risk': heart_risk.astype(int),
    })

    # Save to CSV
    df.to_csv(filename, index=False)

    # Print statistics
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Total records: {len(df)}")
    print(f"  Features: {len(df.columns) - 2}")
    print(f"\nDisease Distribution:")
    print(f"  Diabetes risk cases: {df['diabetes_risk'].sum()} ({100*df['diabetes_risk'].mean():.1f}%)")
    print(f"  Heart disease cases: {df['heart_risk'].sum()} ({100*df['heart_risk'].mean():.1f}%)")
    print(f"\nData saved to: {filename}")

    return df


if __name__ == "__main__":
    generate_sample_data()
