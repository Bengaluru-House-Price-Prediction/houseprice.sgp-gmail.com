# Step 1 — Disable GPU/CUDA BEFORE importing TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Step 2 — Import required libraries
from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
import pandas as pd


# Step 3 — Create Flask app
app = Flask(__name__)


# Step 4 — Load trained ANN model
model = load_model("models/ann_model.keras")


# Step 5 — Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Step 6 — Load feature columns
with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# Step 7 — Get available locations
location_columns = [
    col.replace("location_", "")
    for col in feature_columns
    if col.startswith("location_")
]

locations = sorted(location_columns)


# Step 8 — Home Route
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        print("=== POST REQUEST STARTED ===", flush=True)

        try:
            # Step 9 — Get user inputs
            total_sqft = float(request.form["total_sqft"])
            bath = int(request.form["bath"])
            balcony = int(request.form["balcony"])
            bhk = int(request.form["bhk"])

            # Step 10 — Input validation
            if total_sqft <= 0 or bath <= 0 or balcony < 0 or bhk <= 0:
                return render_template(
                    "index.html",
                    locations=locations,
                    error="Please enter valid property details."
                )

            area_type = request.form["area_type"]
            location = request.form["location"]

            print("=== INPUTS RECEIVED ===", flush=True)

            # Step 11 — Create 189-feature input
            input_data = [0] * len(feature_columns)

            input_data[0] = total_sqft
            input_data[1] = bath
            input_data[2] = balcony
            input_data[3] = bhk


            # Step 12 — Add Area Type
            area_column = "area_type_" + area_type

            if area_column in feature_columns:
                area_index = feature_columns.index(area_column)
                input_data[area_index] = 1


            # Step 13 — Add Location
            location_column = "location_" + location

            if location_column in feature_columns:
                location_index = feature_columns.index(location_column)
                input_data[location_index] = 1

            else:
                if "location_other" in feature_columns:
                    other_index = feature_columns.index("location_other")
                    input_data[other_index] = 1


            # Step 14 — Create DataFrame
            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            print("=== INPUT DATA CREATED ===", flush=True)
            print("INPUT SHAPE:", input_df.shape, flush=True)


            # Step 15 — Scale input
            input_scaled = scaler.transform(input_df)

            print("=== INPUT SCALED ===", flush=True)
            print("SCALED SHAPE:", input_scaled.shape, flush=True)


            # Step 16 — Verify model
            print("MODEL INPUT SHAPE:", model.input_shape, flush=True)
            print("MODEL OUTPUT SHAPE:", model.output_shape, flush=True)


            # Step 17 — Make prediction
            print("=== STARTING PREDICTION ===", flush=True)

            prediction = model(
                input_scaled,
                training=False
            ).numpy()

            print("RAW PREDICTION:", prediction, flush=True)


            # Step 18 — Get predicted price
            predicted_price = float(prediction[0][0])

            print(
                "=== PREDICTION COMPLETED ===",
                predicted_price,
                flush=True
            )


            # Step 19 — Send prediction to HTML
            return render_template(
                "index.html",
                prediction=round(predicted_price, 2),
                locations=locations
            )


        except Exception as e:

            print("=== ERROR OCCURRED ===", flush=True)
            print(str(e), flush=True)

            return render_template(
                "index.html",
                locations=locations,
                error="Prediction failed. Please check the entered values."
            )


    # GET request
    return render_template(
        "index.html",
        locations=locations
    )


# Step 20 — Run Application
if __name__ == "__main__":
    app.run()
