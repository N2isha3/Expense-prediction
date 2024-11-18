import tensorflow as tf 
import pandas as pd
import numpy as np
import pdfplumber
import re
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# Home route to show a welcome page (or basic info)
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html template in the templates folder

# Extract data from PDF
def extract_data_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Process extracted text into a DataFrame
def process_data(text):
    data = []
    pattern = r'(\w{3} \d{1,2}, \d{4})\s+(.*)\s+(DEBIT|CREDIT)\s+₹([\d,]+(?:\.\d{2})?)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        date_str, description, trans_type, amount_str = match
        # Coerce invalid dates to NaT
        date = pd.to_datetime(date_str, errors='coerce')
        amount = float(amount_str.replace(',', ''))
        data.append([date, description, amount])
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=["Date", "Description", "Amount"])
    
    # Drop rows where 'Date' is NaT (invalid date)
    df.dropna(subset=["Date"], inplace=True)

    # Convert 'Amount' to numeric, forcing errors to NaN
    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')

    # Drop rows where 'Amount' is NaN (invalid amounts)
    df.dropna(subset=["Amount"], inplace=True)
    
    return df

# Preprocess data for LSTM
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Amount']])
    seq_length = 30

    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i, 0])
            labels.append(data[i, 0])
        return np.array(sequences), np.array(labels)

    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Build and train the model
def build_and_train_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        text = extract_data_from_pdf(file)
        df = process_data(text)

        if df.empty or df['Amount'].isna().all():
            return jsonify({"error": "No valid data found in the PDF."})

        X, y, scaler = preprocess_data(df)
        model = build_and_train_model(X, y)
        predictions = model.predict(X[-30:])  # Predict for last 30 days as an example
        predictions = scaler.inverse_transform(predictions)

        # Generate plot of actual expenses and predicted expenses
        plt.figure(figsize=(10, 6))

        # Plot actual expenses (blue line)
        plt.plot(df.index, df['Amount'], color='blue', label='Actual Expenses')

        # Generate future time indices for predictions (extending past the actual data)
        prediction_index = pd.date_range(df.index[-1], periods=30, freq='D')  # Adjust 'periods' as needed

        # Plot predicted expenses (red dashed line)
        plt.plot(prediction_index, predictions, color='red', linestyle='--', label='Predicted Expenses')

        # Add titles and labels
        plt.title('Actual vs Predicted Expenses')
        plt.xlabel('Date')
        plt.ylabel('Expense Amount (₹)')
        plt.legend()

        # Save plot to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode image in base64 for displaying on the webpage
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        # Render the table and plot image on the webpage
        return render_template('index.html', table=df.to_html(classes='table table-striped'), plot_img=img_base64)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return a "No Content" response for the favicon request

if __name__ == '__main__':
    app.run(port=5000, debug=True)
