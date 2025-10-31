from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained Random Forest model
with open("carbon_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        energy = float(request.form['Energy_Usage'])
        transport = float(request.form['Transport_Emissions'])
        waste = float(request.form['Waste'])
        water = float(request.form['Water_Usage'])

        # ✅ Make sure these columns match EXACTLY your training dataset
        input_data = pd.DataFrame([[energy, transport, waste, water]], columns=['Energy_Usage', 'Transport_Emissions', 'Waste', 'Water_Usage'])

        # Predict
        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)

        return render_template('index.html', prediction_text=f"🌱 Estimated Carbon Emission: {prediction} kg CO₂")

    except Exception as e:
        # Debug print to console (you can uncomment this during testing)
        print("❌ Prediction Error:", e)
        return render_template('index.html', prediction_text="❌ Unable to predict. Check input values or column names.")

if __name__ == "__main__":
    app.run(debug=True)
