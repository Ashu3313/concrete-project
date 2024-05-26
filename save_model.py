import pickle
import xgboost as xgb

# Load the existing model (assuming it was saved using pickle previously)
model = pickle.load(open('model.pkl', 'rb'))

# Save the model in the JSON format
model.save_model('model.json')

print("Model has been saved in JSON format as 'model.json'")
