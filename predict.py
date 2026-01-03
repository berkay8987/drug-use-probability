import torch
import pandas as pd
from model import DrugRiskANN
import sys

AGE_MAP = {
    '18-24': -0.95197, '25-34': -0.07854, '35-44': 0.49788, 
    '45-54': 1.09449, '55-64': 1.82213, '65+': 2.59171
}

GENDER_MAP = {
    'Female': 0.48246, 'Male': -0.48246
}

EDUCATION_MAP = {
    'Left school before 16 years': -2.43591,
    'Left school at 16 years': -1.73790,
    'Left school at 17 years': -1.43719,
    'Left school at 18 years': -1.22751,
    'Some college or university, no certificate or degree': -0.61113,
    'Professional certificate/ diploma': -0.05921,
    'University degree': 0.45468,
    'Masters degree': 1.16365,
    'Doctorate degree': 1.98437
}

COUNTRY_MAP = {
    'Australia': -0.09765, 'Canada': 0.24923, 'New Zealand': -0.46841,
    'Other': -0.28519, 'Republic of Ireland': 0.21128, 'UK': 0.96082, 'USA': -0.57009
}

ETHNICITY_MAP = {
    'Asian': -0.50212, 'Black': -1.10702, 'Mixed-Black/Asian': 1.90725,
    'Mixed-White/Asian': 0.12600, 'Mixed-White/Black': -0.22166,
    'Other': 0.11440, 'White': -0.31685
}

NSCORE_MAP = {'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5}
ESCORE_MAP = {'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5}
OSCORE_MAP = {'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5}
ASCORE_MAP = {'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5}
CSCORE_MAP = {'Very Low': -2.5, 'Low': -1.0, 'Average': 0.0, 'High': 1.0, 'Very High': 2.5}
IMPULSIVE_MAP = {'Very Low': -2.55, 'Low': -1.37, 'Average': -0.21, 'High': 0.52, 'Very High': 1.86}
SS_MAP = {'Very Low': -2.07, 'Low': -1.18, 'Average': -0.21, 'High': 0.40, 'Very High': 1.22}

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

def get_user_input(prompt, options_map):
    print(f"\n{prompt}")
    options = list(options_map.keys())
    for i, opt in enumerate(options):
        print(f"{i+1}. {opt}")
    
    while True:
        try:
            choice = int(input("Enter choice number: "))
            if 1 <= choice <= len(options):
                return options_map[options[choice-1]]
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")

def predict():
    print("=== Drug Consumption Risk Predictor ===")
    
    age = get_user_input("Select Age Group:", AGE_MAP)
    gender = get_user_input("Select Gender:", GENDER_MAP)
    education = get_user_input("Select Education Level:", EDUCATION_MAP)
    country = get_user_input("Select Country:", COUNTRY_MAP)
    ethnicity = get_user_input("Select Ethnicity:", ETHNICITY_MAP)
    
    print("\n--- Personality Traits (Self-Assessment) ---")
    nscore = get_user_input("Neuroticism (prone to worry/anxiety):", NSCORE_MAP)
    escore = get_user_input("Extraversion (sociable/active):", ESCORE_MAP)
    oscore = get_user_input("Openness to Experience (imaginative/curious):", OSCORE_MAP)
    ascore = get_user_input("Agreeableness (trusting/cooperative):", ASCORE_MAP)
    cscore = get_user_input("Conscientiousness (organized/disciplined):", CSCORE_MAP)
    impulsive = get_user_input("Impulsivity (acting without thinking):", IMPULSIVE_MAP)
    ss = get_user_input("Sensation Seeking (thrill seeking):", SS_MAP)
    
    features = [nscore, escore, oscore, ascore, cscore, impulsive, ss, age, gender, education, country, ethnicity]
    input_tensor = torch.FloatTensor([features])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = DrugRiskANN(num_targets=len(DRUGS)).to(device)
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
    except FileNotFoundError:
        print("\nError: 'best_model.pth' not found. Please run 'train.py' first.")
        return
    except Exception as e:
        print(f"\nError loading model: {e}")
        return

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor) # (1, 19, 7)
        
        probs = torch.nn.functional.softmax(outputs, dim=2) # (1, 19, 7)
        
        max_probs, predicted = torch.max(probs, 2) # (1, 19)
        
    print("\n=== Predictions ===")
    print(f"{'Drug':<30} | {'Predicted Usage Risk':<30} | {'Confidence Rate'}")
    print("-" * 80)
    
    results = []
    for i, drug in enumerate(DRUGS):
        risk_level = predicted[0][i].item()
        risk_desc = CLASS_MAP[risk_level]
        probability = max_probs[0][i].item() * 100
        results.append((drug, risk_desc, probability))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    for drug, risk_desc, probability in results:
        print(f"{drug:<30} | {risk_desc:<30} | {probability:.2f}%")

    print("\n" + "="*80)
    print("DISCLAIMER: Interpreting the Results")
    print("-" * 80)
    print("These predictions combine a usage risk category with a confidence score.")
    print("For example, a result of 'Crack | Used in Last Week | 52%' indicates that")
    print("the model estimates a 52% probability that the individual falls into the")
    print("'Used in Last Week' category for Crack consumption.")
    print("These are statistical estimates based on personality profiles and should")
    print("not be taken as definitive proof of drug use.")
    print("="*80 + "\n")

if __name__ == "__main__":
    predict()
