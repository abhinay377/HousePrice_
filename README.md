# 🏠 House Price Prediction — ML Internship Project

## Project Structure
```
house-price-prediction/
├── house_price_prediction.ipynb   # Jupyter Notebook (full ML pipeline)
├── app.py                         # Flask backend API
├── static/
│   └── index.html                 # Frontend UI
├── model/                         # Auto-generated after running notebook
│   ├── house_price_model.pkl
│   ├── encoders.pkl
│   └── metadata.json
├── requirements.txt
└── README.md
```

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook
Open and run all cells in `house_price_prediction.ipynb`.  
This trains the model and saves files to the `model/` folder.

### Step 3 — Start the Flask server
```bash
python app.py
```

### Step 4 — Open the web app
Visit: http://localhost:5000

## ML Pipeline Summary
| Step | Description |
|------|-------------|
| Data | 2000-row synthetic Hyderabad property dataset |
| EDA  | Distribution plots, correlation heatmap, neighborhood analysis |
| Models | Linear Regression, Random Forest, Gradient Boosting |
| Best Model | Random Forest (highest R²) |
| API | Flask REST endpoint `/api/predict` |
| Frontend | Vanilla HTML/CSS/JS with live predictions |

## API Endpoints
- `GET /api/metadata` — returns dropdown options and model R² score
- `POST /api/predict` — accepts property features, returns predicted price

## Tech Stack
Python · scikit-learn · pandas · Flask · HTML/CSS/JS
