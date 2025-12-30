from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests
from geopy.geocoders import Nominatim

app = Flask(__name__)
# ===== LOAD MODELS =====
yield_model = joblib.load("models/yield_model.pkl")
yield_scaler = joblib.load("models/yield_scaler.pkl")
yield_columns = joblib.load("models/yield_columns.pkl")

fert_model = joblib.load("models/fertilizer_model.pkl")
fert_encoders = joblib.load("models/fertilizer_encoders.pkl")
fert_target = joblib.load("models/fertilizer_target_encoder.pkl")

WEATHER_API_KEY = "74d02e94175d7d2861e3de10df92f79b"

def get_weather(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        temperature = data["main"]["temp"]
        rainfall = data.get("rain", {}).get("1h", 0)

        return temperature, rainfall

    except Exception as e:
        print("Weather API error:", e)
        return 30, 0   # fallback
    

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/yield_form")
def yield_form():
    return render_template("yield_form.html")

@app.route("/yield", methods=["POST"])
def yield_predict():
    form = request.form

    location = form["location"]
    crop = form["crop_type"]

    # Convert location → lat/lon
    geolocator = Nominatim(user_agent="crop-app")
    loc = geolocator.geocode(f"{location}, India")

    if not loc:
        return "Invalid location"

    lat, lon = loc.latitude, loc.longitude

    # Fetch weather automatically
    temperature, rainfall = get_weather(lat, lon)

    # Build input data
    input_data = {
        "latitude": lat,
        "longitude": lon,
        "temperature": temperature,
        "rainfall": rainfall,
        "soil_moisture": float(form["soil_moisture"]),
        "NDVI": float(form["ndvi"]),
        "GNDVI": float(form["gndvi"]),
        "NDWI": float(form["ndwi"]),
        "SAVI": float(form["savi"]),
        "crop_type": crop
    }

    df = pd.DataFrame([input_data])

    # Encode crop
    df = pd.get_dummies(df, columns=["crop_type"])
    df = df.reindex(columns=yield_columns, fill_value=0)

    # Scale + Predict
    scaled = yield_scaler.transform(df)
    prediction = yield_model.predict(scaled)[0]

    return render_template(
        "yield_result.html",
        yield_value=round(prediction, 2),
        location=loc.address,
        crop=crop,
        inputs=input_data
    )

@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")

@app.route("/fertilizer_predict", methods=["POST"])
def fertilizer_predict():
    form = request.form

    input_data = {
        "Soil_Type": form["soil_type"],
        "Soil_pH": float(form["soil_ph"]),
        "Soil_Moisture": float(form["soil_moisture"]),
        "Nitrogen_Level": float(form["nitrogen"]),
        "Phosphorus_Level": float(form["phosphorus"]),
        "Potassium_Level": float(form["potassium"]),
        "Temperature": float(form["temperature"]),
        "Humidity": float(form["humidity"]),
        "Rainfall": float(form["rainfall"]),
        "Crop_Type": form["crop_type"],
        "Previous_Crop": form["previous_crop"],
        "Region": form["region"],
        "Fertilizer_Used_Last_Season": form["last_fert"],
        "Yield_Last_Season": float(form["last_yield"])
    }

    df = pd.DataFrame([input_data])

    for col, enc in fert_encoders.items():
        df[col] = df[col].apply(lambda x: x if x in enc.classes_ else enc.classes_[0])
        df[col] = enc.transform(df[col])

    fertilizer = fert_target.inverse_transform(fert_model.predict(df))[0]

    return render_template("fertilizer_result.html", fertilizer=fertilizer)

if __name__ == "__main__":
    app.run(debug=True)
