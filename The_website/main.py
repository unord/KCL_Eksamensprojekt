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
@app.route('/<string:page>/', methods=["GET", "POST"])
def home(page=None):
    if page == 'Page2':
        # Handle the id form submission here if needed
        return render_template('Page2.html')  # Render Page2.html
    elif page == 'Page3':
        return render_template('Page3.html')  # Render Page3.html
    else:
        return render_template('home.html')  # Render home.html if no specific page requested


# Start the Flask application
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
