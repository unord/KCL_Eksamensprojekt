from flask import *
import numpy as np
import pickle

# Initialize Flask application
app = Flask(__name__)
# Set secret key for Flask session
app.config['SECRET_KEY'] = 'HTX123'  

# Load the pre-trained model for future predictions
### model = pickle.load(open('StarID/flask_penguins/static/models/star_model' , 'rb'))

# Define the default route that serves the homepage
@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('home.html')

# Define the route to handle star identification
@app.route('/id/', methods=["GET", "POST"])
def id():
    if request.method == "POST":
        print(request.form)

        arr = []  # Initialize an empty list to store processed form values
        for key, value in request.form.items():
            if key == "Color":
                value = value.lower()  # Convert to lowercase for consistency
                
                # Map certain color names to standardized names
                color_map = {
                    'blue white': 'blue-white',
                    'yellow-white': 'white-yellow',
                    'yellowish white': 'yellowish'
                }
                value = color_map.get(value, value)  # Get the standardized name, if it exists

                # Convert color names to numerical values for the model
                color_encoding = {
                    'red': 0, 
                    'blue-white': 1, 
                    'white': 2, 
                    'blue': 3, 
                    'yellowish': 4, 
                    'pale-yellow-orange': 5, 
                    'whitish': 6, 
                    'white-yellow': 7, 
                    'orange': 8, 
                    'orange-red': 9
                }
                value = color_encoding[value]
            elif key == "Sclass":
                
                # Handle spectral class input and encode it
                spectral_class_encoding = {
                    'O': 0,
                    'B': 1,
                    'A': 2,
                    'F': 3,
                    'G': 4,
                    'K': 5,
                    'M': 6
                }
                value = spectral_class_encoding[value]
            elif key == "id_star_button":
                continue  # If the input is the button, skip it
            else:
                # For other inputs, convert them to float values
                value = float(value)

            arr.append(value)  # Add the processed value to the list

        # Use the model to predict the star type based on the input values
        prediction = model.predict(np.array(arr).reshape(1, -1))

        name_map = {
                    0: 'Red Dwarf',
                    1: 'Brown Dwarf',
                    2: 'White Dwarf',
                    3: 'Main Sequence',
                    4: 'Super Giant',
                    5: 'Hyper Giant'
        }
        print(prediction)
        prediction = name_map[prediction[0]]

    return jsonify({"prediction": str(prediction)})

# Start the Flask application
if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0', port=5000)
