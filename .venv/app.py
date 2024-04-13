from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import gpxpy
from shapely.geometry import Polygon
import requests #weather api

#import model
model1= pickle.load(open('model.pkl','rb'))
ms=pickle.load(open('minmaxscaler.pkl','rb'))
sc=pickle.load(open('standscaler.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process_gpx', methods=['POST'])
def process_gpx():
    if 'gpx_file' not in request.files:
        return "No GPX file found", 400

    gpx_file = request.files['gpx_file']

    if gpx_file.filename == '':
        return "No selected file", 400

    # Save the uploaded GPX file temporarily
    gpx_file_path = 'temp.gpx'
    gpx_file.save(gpx_file_path)

    # Calculate area from the uploaded GPX file
    area = calculate_area(gpx_file_path)

    # Extract coordinates from the GPX file
    latitude, longitude = extract_coordinates(gpx_file_path)

    # Get weather data based on coordinates
    weather = get_weather_data(latitude, longitude)

    # Get soil data based on coordinates
    nitrogen, phosphorus, potassium = get_soil_data(latitude, longitude)

    return render_template("index.html", area=area, weather=weather, nitrogen=nitrogen, phosphorus=phosphorus,potassium=potassium)

def calculate_area(gpx_file_path):
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)

    total_area = 0
    for track in gpx.tracks:
        for segment in track.segments:
            points = [(point.longitude, point.latitude) for point in segment.points]
            polygon = Polygon(points)
            total_area += polygon.area

    # Convert area from square meters to hectares
    area_hectares = total_area / 10000

    return area_hectares


def get_soil_data(latitude, longitude):
    url = f"https://rest.soilgrids.org/query?lon={longitude}&lat={latitude}"

    try:
        # Send request to SoilGrids API
        response = requests.get(url)
        data = response.json()

        # Extract soil data (NPK values) from the response
        nitrogen = data['properties']['nitrogen']['mean']
        phosphorus = data['properties']['phh2o']['mean']
        potassium = data['properties']['phh2o']['mean']

        return nitrogen, phosphorus, potassium
    except Exception as e:
        print("Error retrieving soil data:", e)
        return None, None, None

def extract_coordinates(gpx_file_path):
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)

    # Assuming the GPX file contains a single track with multiple segments
    track = gpx.tracks[0]
    segments = track.segments

    # Extracting coordinates from the first segment
    points = segments[0].points
    latitude = points[0].latitude
    longitude = points[0].longitude

    return latitude, longitude

def get_weather_data(latitude, longitude):
    api_key = "ac7efd4ee28ab28391d197d276453afc"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    # Check if the 'main' key is present in the API response
    if 'main' not in data:
        return None, None, None

    # Access temperature, humidity, and rainfall if available
    temperature = data['main'].get('temp', None)
    humidity = data['main'].get('humidity', None)
    rainfall = data.get('rain', {}).get('1h', 0) if 'rain' in data else 0  # Handle missing rain data

    return temperature, humidity, rainfall

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model1.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)

if __name__=="__main__":
    app.run(debug=True)
