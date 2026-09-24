#Step 1 — Import required libraries
from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

#Step 2 - creating flask app
app = Flask(__name__)

model = load_model("models/ann_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

    # Step 10 : Get available locations

    location_columns = [
    col.replace("location_", "")
    for col in feature_columns
    if col.startswith("location_")
    ]

    locations = sorted(location_columns)
    
#Step 3 : Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        
          # Step 5 : Get User Inputs

        # Step 12 : Input Validation
        print("=== POST REQUEST STARTED ===", flush=True)

        total_sqft = float(request.form["total_sqft"])
        bath = int(request.form["bath"])
        balcony = int(request.form["balcony"])
        bhk = int(request.form["bhk"])

        if total_sqft <= 0 or bath <= 0 or balcony < 0 or bhk <= 0:
            return render_template(
            "index.html",
            locations=locations,
            error="Please enter valid property details."
        )

        area_type = request.form["area_type"]
        location = request.form["location"]

        # Step 6 : Create 189-feature input
        input_data = [0] * len(feature_columns)

        input_data[0] = total_sqft
        input_data[1] = bath
        input_data[2] = balcony
        input_data[3] = bhk

        # Step 6.1 : Add Area Type
        # Step 6.1 : Add Area Type
        area_column = "area_type_" + area_type

        if area_column in feature_columns:
            area_index = feature_columns.index(area_column)
            input_data[area_index] = 1
        else:
            # Area type is not available in the trained feature columns
            pass
        # Step 6.2 : Add Location
        location_column = "location_" + location

        if location_column in feature_columns:
            location_index = feature_columns.index(location_column)
            input_data[location_index] = 1
        else:
            input_data[feature_columns.index("location_other")] = 1
            
        # Step 7 : Scale the input
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        input_scaled = scaler.transform(input_df)

        print("=== BEFORE PREDICTION ===", flush=True)
        print("MODEL INPUT SHAPE:", input_scaled.shape, flush=True)
        print("MODEL EXPECTED SHAPE:", model.input_shape, flush=True)
        print("MODEL OUTPUT SHAPE:", model.output_shape, flush=True)
        print("MODEL LOADED SUCCESSFULLY", flush=True)

        prediction = model.predict(input_scaled, verbose=0)

        print("RAW PREDICTION:", prediction, flush=True)

        predicted_price = float(prediction[0][0])

        print("=== PREDICTION COMPLETED ===", predicted_price, flush=True)

        print("=== PREDICTION COMPLETED ===", predicted_price, flush=True)
        
        print("Prediction completed:", predicted_price)
        print("=== PREDICTION COMPLETED ===", predicted_price, flush=True)
        # Step 9 : Send prediction to HTML

        return render_template(
        "index.html",
        prediction=round(predicted_price, 2),
        locations=locations
        )   
    return render_template(
    "index.html",
    locations=locations
    )
#Step 4 : Run Application
if __name__ == "__main__":
    app.run()
