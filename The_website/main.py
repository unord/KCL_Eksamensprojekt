"""
File: main.py
Description: 
This script serves as the backend for a Flask web application designed to predict the type of injury an individual might sustain in a traffic accident based on various input parameters. It uses a pre-trained decision tree model to make predictions.

The application comprises the following main components:
1. 'Flask Setup': Initializes the Flask application and configures the secret key for session management.
2. 'Model Loading': Loads the pre-trained decision tree model from a .joblib file.
3. 'Label Encoding': Defines label encoders for transforming categorical input data into numerical format suitable for model prediction.
4. 'Routes': 
   - Home Route ('/'): Renders the home page and handles navigation to other pages.
   - Prediction Route ('/predict_injury'): Accepts POST requests with input data, processes the data, makes predictions using the model, and returns the prediction.
   - More Info Route ('/more_info'): Provides additional information based on the user's inputs and prediction results, including alternative regions and vehicle types with the best and worst predicted outcomes.

Functions:
- 'home(page=None)': Renders the appropriate template based on the requested page.
- 'predict_injury()': Processes input data, makes a prediction using the loaded model, and returns the result as a JSON formatted response.
- 'more_info()': Calculates alternative outcomes based on different regions and vehicle types, and renders the 'more_info' template with these details.

Usage:
Run this script to start the Flask web server. The server will be accessible at 'http://0.0.0.0:5000'.

Dependencies:
- Flask
- numpy
- pandas
- joblib
- scikit-learn
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the dtree model
model = load('The_website/static/ml_model/decision_tree_model.joblib')

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'HTX123' # Secret Key

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
    """
    Handles routing for the home page and other static pages of the web application.

    The home function is responsible for rendering different templates based on the URL path provided. It supports multiple routes to render specific pages within the application. The function uses Flask's 'render_template' method to serve HTML templates.

    Routes:
    - '/': Renders the home page.
    - '/Page2/': Renders the Page2 template.
    - '/Page3/': Renders the Page3 template.

    Parameters:
    - page (str, optional): The specific page to be rendered. Default is None, which renders the home page.

    Returns:
    - str: Rendered HTML template for the requested page.

    Example:
    - Accessing the root URL ('/') will render 'home.html'.
    - Accessing '/Page2/' will render 'Page2.html'.
    """
    if page == 'Page2':
        return render_template('Page2.html')
    elif page == 'Page3':
        return render_template('Page3.html')
    else:
        return render_template('home.html')


@app.route('/predict_injury', methods=['POST'])
def predict_injury():
    """
    Handles the prediction of injury severity based on user input data.

    The predict_injury function receives JSON data from a POST request, processes the data to make predictions using our trained decision tree model, and returns the prediction result as a JSON response. 
    The function also stores the prediction and input data in the session for later use.

    Method:
    - POST

    Request Data (JSON):
    - gender (str): The gender of the user.
    - age (str): The age range of the user.
    - area (str): The area/region where the user is located.
    - vehicle (str): The type of vehicle the user is using.

    Returns:
    - JSON: A JSON response containing the predicted class label.

    Example:
    - Input: {'gender': 'Mænd', 'age': '18-24 År', 'area': 'Region Hovedstaden', 'vehicle': 'Almindelig personbil'}
    - Output: {'prediction': 'Lettere tilskadekomne'}
    """
    data = request.get_json(force=True)
    print("Received data:", data)  # Debug print for received data

    gender = data['gender']
    age = data['age']
    area = data['area'].title()  # Ensures consistent capitalization from Page2
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
    
    print("Encoded input data:", input_data)  # Debug print for encoded input data
    
    # Makes prediction using our trained model
    prediction = model.predict(input_data)
    
    # Gets the predicted class label
    predicted_class = prediction[0]
    print("Prediction:", predicted_class)  # Debug print for prediction
    
    # Stores the prediction result and user inputs in 'session'
    session['prediction'] = predicted_class
    session['inputs'] = data
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': predicted_class})


@app.route('/more_info', methods=['GET'])
def more_info():
    """
    Provides extra information and alternative predictions based on the original user inputs.

    The more_info function generates additional insights by comparing the initial prediction with alternative scenarios.
    It uses the user's input stored in the session to calculate the best and worst regions and vehicles in terms of predicted
    injury severity. The function then renders a template: "more_info" with these insights.

    Method:
    - GET

    Session Data:
    - inputs (dict): User inputs including gender, age, area, and vehicle.

    Returns:
    - Rendered HTML template with detailed information and alternative predictions.

    Example:
    - If the user has provided inputs, this function will use the session data to calculate and display best and worst regions
      and same with vehicles for the given user.
    """
    # Check if user inputs are stored in the session
    if 'inputs' not in session:
        # If no inputs are found, redirect to the home page
        return redirect(url_for('home'))
    
    # Retrieve previous user inputs from the session
    inputs = session['inputs']
    gender = inputs['gender']
    age = inputs['age']

    # Map for prediction values to numerical scores
    prediction_mapping = {
        'Lettere tilskadekomne': 1,
        'Alvorligt tilskadekomne': 2,
        'Dræbte': 3
    }

    # Prepare data to calculate alternatives for all combinations of regions and vehicle types
    regions = label_encoders['Area'].classes_  # List of all possible regions
    vehicles = label_encoders['Type of Vehicle'].classes_  # List of all possible vehicle types
    results = []  # List to store prediction results

    # Iterate over all combinations of regions and vehicle types
    for r in regions:
        for v in vehicles:
            # Create a DataFrame for the current combination of input features
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Area': [r],
                'Type of Vehicle': [v]
            })
            # Encode the input data using the defined label encoders
            for column, le in label_encoders.items():
                input_data[column] = le.transform(input_data[column])
            # Make a prediction for the current combination
            prediction = model.predict(input_data)[0]
            # Append the result to the results list
            results.append({
                'region': r,
                'vehicle': v,
                'prediction': prediction,
                'prediction_score': prediction_mapping[prediction] # Get injury type str
            })

    # Calculate averages for each region and vehicle type
    region_avg = {r: [] for r in regions}  # Initialize dictionary to store region averages
    vehicle_avg = {v: [] for v in vehicles}  # Initialize dictionary to store vehicle averages

    # Populate the dictionaries with prediction scores
    for result in results:
        region_avg[result['region']].append(result['prediction_score'])
        vehicle_avg[result['vehicle']].append(result['prediction_score'])

    # Calculate the best and worst regions and vehicles based on average prediction scores. / Lowest average injury types for all vehicles in all regions
    best_region = min(region_avg, key=lambda k: sum(region_avg[k])/len(region_avg[k]))
    worst_region = max(region_avg, key=lambda k: sum(region_avg[k])/len(region_avg[k]))
    best_vehicle = min(vehicle_avg, key=lambda k: sum(vehicle_avg[k])/len(vehicle_avg[k]))
    worst_vehicle = max(vehicle_avg, key=lambda k: sum(vehicle_avg[k])/len(vehicle_avg[k]))

    # Group the prediction results by region for easier display in the template
    grouped_results = {}
    for result in results:
        if result['region'] not in grouped_results:
            grouped_results[result['region']] = []
        grouped_results[result['region']].append(result)

    # Render the more_info.html template with the calculated 'insights' and grouped results
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
