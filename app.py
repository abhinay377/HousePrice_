from flask import Flask, request, jsonify, send_from_directory
import pickle, json, numpy as np, pandas as pd, os

app = Flask(__name__, static_folder='static')

# ── Load model & encoders ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

with open(f'{MODEL_DIR}/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'{MODEL_DIR}/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open(f'{MODEL_DIR}/metadata.json') as f:
    metadata = json.load(f)

FEATURE_COLS = metadata['feature_cols']

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/metadata')
def get_metadata():
    return jsonify(metadata)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()

        # Raw inputs
        area      = float(body['area'])
        bedrooms  = int(body['bedrooms'])
        bathrooms = int(body['bathrooms'])
        age       = int(body['age'])
        parking   = int(body['parking'])
        floor     = int(body.get('floor', 0))
        nbr       = body['neighborhood']
        ptype     = body['property_type']
        furnish   = body['furnishing']
        condition = body['condition']

        # Encode categoricals
        nbr_enc  = int(encoders['Neighborhood'].transform([nbr])[0])
        pt_enc   = int(encoders['Property_Type'].transform([ptype])[0])
        fn_enc   = int(encoders['Furnishing'].transform([furnish])[0])
        cond_enc = int(encoders['Condition'].transform([condition])[0])

        # Engineered features
        room_ratio  = bathrooms / max(bedrooms, 1)
        total_rooms = bedrooms + bathrooms

        row = {
            'Area_sqft': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
            'Age_years': age, 'Parking_spots': parking, 'Floor': floor,
            'Room_ratio': room_ratio, 'Total_rooms': total_rooms,
            'Neighborhood_enc': nbr_enc, 'Property_Type_enc': pt_enc,
            'Furnishing_enc': fn_enc, 'Condition_enc': cond_enc
        }
        X = pd.DataFrame([[row[fc] for fc in FEATURE_COLS]], columns=FEATURE_COLS)
        price = float(model.predict(X)[0])

        # Confidence range ±8%
        return jsonify({
            'predicted_price': round(price, -4),
            'range_low':  round(price * 0.92, -4),
            'range_high': round(price * 1.08, -4),
            'price_per_sqft': round(price / area),
            'model_r2': metadata['model_r2']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
