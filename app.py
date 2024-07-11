from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('RanFor.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract the features from the form data
    features = [
        data['person_age'],
        data['person_income'],
        data['person_emp_length'],
        data['loan_amnt'],
        data['loan_int_rate'],
        data['loan_percent_income'],
        data['cb_person_cred_hist_length'],
        data['person_home_ownership'],
        data['loan_intent'],
        data['loan_grade'],
        data['cb_person_default_on_file']
    ]

    # Preprocess the features if necessary (e.g., convert to numerical values, scale, etc.)
    # This depends on how your model was trained
    # For example:
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict_proba(features)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
