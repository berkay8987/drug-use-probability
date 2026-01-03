import sys
import os
import torch
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DrugRiskANN

app = Flask(__name__)

DRUGS = [
    'Alcohol', 'Amphetamines', 'Amyl Nitrite', 'Benzodiazepines', 'Caffeine', 'Cannabis', 'Chocolate', 'Cocaine', 
    'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legal Highs', 'LSD', 'Methadone', 
    'Magic Mushrooms', 'Nicotine', 'Semeron (Fictitious)', 'Volatile Substance Abuse'
]

CLASS_MAP = {
    0: "Never Used",
    1: "Used over a Decade Ago",
    2: "Used in Last Decade",
    3: "Used in Last Year",
    4: "Used in Last Month",
    5: "Used in Last Week",
    6: "Used in Last Day"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
        model = DrugRiskANN(num_targets=len(DRUGS)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    
    try:
        features = [
            float(data['nscore']), float(data['escore']), float(data['oscore']), 
            float(data['ascore']), float(data['cscore']), float(data['impulsive']), 
            float(data['ss']), float(data['age']), float(data['gender']), 
            float(data['education']), float(data['country']), float(data['ethnicity'])
        ]
        
        input_tensor = torch.FloatTensor([features]).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=2)
            max_probs, predicted = torch.max(probs, 2)
            
        results = []
        for i, drug in enumerate(DRUGS):
            risk_level = predicted[0][i].item()
            risk_desc = CLASS_MAP[risk_level]
            probability = max_probs[0][i].item() * 100
            
            results.append({
                'drug': drug,
                'risk': risk_desc,
                'confidence': round(probability, 2)
            })
            
        results.sort(key=lambda x: x['confidence'], reverse=True)
            
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
