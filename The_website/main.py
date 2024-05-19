from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load('The_website/static/ml_model/decision_tree_model.joblib')

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'HTX123'

# Label encoders
label_encoders = {
    'Gender': LabelEncoder().fit(['Mænd', 'Kvinder']),
    'Age': LabelEncoder().fit(['0-17 År', '18-24 År', '25-44 År', '45-64 År', '65 År og derover']),
    'Area': LabelEncoder().fit(['Region Hovedstaden', 'Region Sjælland', 'Region Syddanmark', 'Region Midtjylland', 'Region Nordjylland']),
    'Type of Vehicle': LabelEncoder().fit([
        'Almindelig personbil', 'Taxi', 'Køretøj 0-3.500 kg under udrykning', 'Varebil 0-3.500 kg.', 'Lastbil over 3.500 kg.',
        'Bus', 'Motorcykel', 'Knallert 45', 'Knallert', 'Cykel', 'Fodgænger', 'Andre'
    ])
}

@app.route('/', methods=["GET", "POST"])
@app.route('/<string:page>/', methods=["GET", "POST"])
def home(page=None):
    if page == 'Page2':
        return render_template('Page2.html')
    elif page == 'Page3':
        return render_template('Page3.html')
    elif page == 'predict':
        return render_template('predict.html')
    else:
        return render_template('home.html')

@app.route('/predict_injury', methods=['POST'])
def predict_injury():
    data = request.get_json(force=True)
    print("Received data:", data)  # Debug print
    gender = data['gender']
    age = data['age']
    area = data['area'].title()  # Ensure consistent capitalization
    vehicle = data['vehicle']

    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Area': [area],
        'Type of Vehicle': [vehicle]
    })
    
    # Encode the input data using the label encoders
    for column, le in label_encoders.items():
        input_data[column] = le.transform(input_data[column])
    
    print("Encoded input data:", input_data)  # Debug print
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get the predicted class label
    predicted_class = prediction[0]
    print("Prediction:", predicted_class)  # Debug print
    
    # Store the prediction result and user inputs in session
    session['prediction'] = predicted_class
    session['inputs'] = data
    
    return jsonify({'prediction': predicted_class})

@app.route('/more_info', methods=['GET'])
def more_info():
    if 'inputs' not in session:
        return redirect(url_for('home'))
    
    inputs = session['inputs']
    gender = inputs['gender']
    age = inputs['age']
    region = inputs['area'].title()
    vehicle = inputs['vehicle']

    # Define a mapping for prediction values to numerical scores
    prediction_mapping = {
        'Lettere tilskadekomne': 1,
        'Alvorligt tilskadekomne': 2,
        'Dræbte': 3
    }

    # Prepare data to calculate alternatives
    regions = label_encoders['Area'].classes_
    vehicles = label_encoders['Type of Vehicle'].classes_
    results = []

    for r in regions:
        for v in vehicles:
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Area': [r],
                'Type of Vehicle': [v]
            })
            for column, le in label_encoders.items():
                input_data[column] = le.transform(input_data[column])
            prediction = model.predict(input_data)[0]
            results.append({
                'region': r,
                'vehicle': v,
                'prediction': prediction,
                'prediction_score': prediction_mapping[prediction]
            })

    # Calculate averages
    region_avg = {r: [] for r in regions}
    vehicle_avg = {v: [] for v in vehicles}

    for result in results:
        region_avg[result['region']].append(result['prediction_score'])
        vehicle_avg[result['vehicle']].append(result['prediction_score'])

    best_region = min(region_avg, key=lambda k: sum(region_avg[k])/len(region_avg[k]))
    worst_region = max(region_avg, key=lambda k: sum(region_avg[k])/len(region_avg[k]))
    best_vehicle = min(vehicle_avg, key=lambda k: sum(vehicle_avg[k])/len(vehicle_avg[k]))
    worst_vehicle = max(vehicle_avg, key=lambda k: sum(vehicle_avg[k])/len(vehicle_avg[k]))

    # Group results by region for the template
    grouped_results = {}
    for result in results:
        if result['region'] not in grouped_results:
            grouped_results[result['region']] = []
        grouped_results[result['region']].append(result)

    return render_template('more_info.html', 
                           best_region=best_region, 
                           worst_region=worst_region, 
                           best_vehicle=best_vehicle, 
                           worst_vehicle=worst_vehicle,
                           grouped_results=grouped_results)

# Start the Flask application
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
