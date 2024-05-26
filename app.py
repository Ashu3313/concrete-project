from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb

# Load the model from the JSON file
model = xgb.Booster()
model.load_model('model.json')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    binder = float(request.form['binder'])
    slag = float(request.form['slag'])
    temp = float(request.form['temp'])
    water = float(request.form['water'])
    superplasticizer = float(request.form['superplasticizer'])
    courseAggregate = float(request.form['courseAggregate'])
    fineaggregate = float(request.form['fineaggregate'])
    age = int(request.form['age'])

    # Print input features to debug
    print("Input features:")
    print(f"Binder: {binder}, Slag: {slag}, Curing_Temperature: {temp}, Water: {water}, Superplasticizer: {superplasticizer}, Course Aggregate: {courseAggregate}, Fine Aggregate: {fineaggregate}, Age: {age}")

    # Transform input features
    features = np.array([binder, slag, temp, water, superplasticizer, courseAggregate, fineaggregate, age]).reshape(1, -1)
    
    # Print features array to debug
    print("Features array:", features)
    
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(features)
    
    # Make a prediction
    prediction = model.predict(dmatrix)
    
    # Print prediction to debug
    print("Prediction:", prediction)
    
    return render_template('index.html', strength=prediction[0])

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
