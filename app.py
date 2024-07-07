from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
from io import BytesIO
app = Flask(__name__)

# Load the model
model = pickle.load(open('Loan_appr_cat.sav', 'rb'))

prediction_map = {
    0: 'P1',
    1: 'P2',
    2: 'P3',
    3: 'P4'
}

# Home page
@app.route('/')
def home():
    return render_template('pp.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        df = pd.read_excel(file)
        predictions = []
        for index, row in df.iloc[1:].iterrows():
            data = row.values.reshape(1, -1)
            prediction = model.predict(data)[0]
            mapped_prediction = prediction_map.get(prediction, 'Unknown')
            predictions.append(mapped_prediction)
            #predictions.append(int(prediction))
        return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run()

