from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('classifier_model.pkl', 'rb'))

# Feature order (VERY IMPORTANT - same as training)
feature_columns = [
    'pH','EC','OrganicCarbon','Nitrogen','Phosphorus','Potassium','Moisture','CEC',

    # Districts
    'District_Ariyalur','District_Coimbatore','District_Cuddalore',
    'District_Dharmapuri','District_Dindigul','District_Erode',
    'District_Kanchipuram','District_Karur','District_Krishnagiri',
    'District_Madurai','District_Nagapattinam','District_Namakkal',
    'District_Nilgiris','District_Perambalur','District_Pudukkottai',
    'District_Ramanathapuram','District_Salem','District_Sivaganga',
    'District_Thanjavur','District_Theni','District_Thoothukudi',
    'District_Tiruchirappalli','District_Tirunelveli','District_Tiruvallur',
    'District_Tiruvarur','District_Vellore','District_Villupuram',
    'District_Virudhunagar',

    # Texture
    'Texture_Clay','Texture_Clay Loam','Texture_Loam',
    'Texture_Sandy','Texture_Sandy Loam'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Numeric inputs
        data = {
            'pH': float(request.form['pH']),
            'EC': float(request.form['EC']),
            'OrganicCarbon': float(request.form['OrganicCarbon']),
            'Nitrogen': float(request.form['Nitrogen']),
            'Phosphorus': float(request.form['Phosphorus']),
            'Potassium': float(request.form['Potassium']),
            'Moisture': float(request.form['Moisture']),
            'CEC': float(request.form['CEC']),
        }

        # 2. Initialize all districts = 0
        for col in feature_columns:
            if "District_" in col:
                data[col] = 0

        # Selected district = 1
        selected_district = "District_" + request.form['district']
        data[selected_district] = 1

        # 3. Initialize all textures = 0
        for col in feature_columns:
            if "Texture_" in col:
                data[col] = 0

        # Selected texture = 1
        selected_texture = "Texture_" + request.form['texture']
        data[selected_texture] = 1

        # 4. Arrange in correct order
        input_data = [data[col] for col in feature_columns]
        input_array = np.array([input_data])

       # Prediction
        classes = ['Cotton','Groundnut','Maize','Millet','Rice','Sugarcane','Turmeric']
        pred_index = model.predict(input_array)[0]
        prediction = classes[int(pred_index)]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)