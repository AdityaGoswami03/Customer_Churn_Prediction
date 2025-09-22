import pandas as pd
from flask import Flask, request, render_template
import pickle
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Load model
model = pickle.load(open("model.sav", "rb"))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/single', methods=['GET', 'POST'])
def single_input():
    if request.method == 'POST':
        try:
            # Get form input
            input_data = {
                'SeniorCitizen': int(request.form['query1']),
                'MonthlyCharges': float(request.form['query2']),
                'TotalCharges': float(request.form['query3']),
                'gender': request.form['query4'],
                'Partner': request.form['query5'],
                'Dependents': request.form['query6'],
                'PhoneService': request.form['query7'],
                'MultipleLines': request.form['query8'],
                'InternetService': request.form['query9'],
                'OnlineSecurity': request.form['query10'],
                'OnlineBackup': request.form['query11'],
                'DeviceProtection': request.form['query12'],
                'TechSupport': request.form['query13'],
                'StreamingTV': request.form['query14'],
                'StreamingMovies': request.form['query15'],
                'Contract': request.form['query16'],
                'PaperlessBilling': request.form['query17'],
                'PaymentMethod': request.form['query18'],
                'tenure': int(request.form['query19'])
            }

            # Preprocessing
            new_df = pd.DataFrame([input_data])
            labels = [f"{i}-{i+11}" for i in range(1, 72, 12)]
            new_df['tenure_group'] = pd.cut(new_df['tenure'], bins=range(1, 80, 12), right=False, labels=labels)
            new_df.drop('tenure', axis=1, inplace=True)

            categorical_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
            ]
            dummy_df = pd.get_dummies(new_df[categorical_cols])
            expected_columns = model.feature_names_in_
            for col in expected_columns:
                if col not in dummy_df.columns:
                    dummy_df[col] = 0
            dummy_df = dummy_df[expected_columns]

            # Prediction
            prediction = model.predict(dummy_df)[0]
            probability = model.predict_proba(dummy_df)[0, 1]

            if prediction == 1:
                output1 = "This customer is likely to churn!"
                output2 = f"{probability * 100:.1f}%"
            else:
                output1 = "This customer is likely to stay."
                output2 = f"{(1 - probability) * 100:.1f}%"

            return render_template(
                "single_input.html",
                output1=output1,
                output2=output2,
                **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)}
            )
        except Exception as e:
            return render_template("single_input.html", output1="Error", output2=str(e))

    return render_template("single_input.html")

@app.route('/batch', methods=['GET', 'POST'])
def batch_input():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template("batch_input.html", output=["No file selected"])
            
        file = request.files['file']
        
        # Validate file
        if file.filename == '':
            return render_template("batch_input.html", output=["No file selected"])
            
        if not allowed_file(file.filename):
            return render_template("batch_input.html", output=["Invalid file type. Only CSV files are allowed."])

        try:
            # Read and process the file
            df = pd.read_csv(file)
            
            # Validate required columns
            required_columns = ['customerID', 'tenure']  # Add all your required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return render_template("batch_input.html", 
                                    output=[f"Missing required columns: {', '.join(missing_cols)}"])

            # Preprocessing
            labels = [f"{i}-{i+11}" for i in range(1, 72, 12)]
            df['tenure_group'] = pd.cut(df['tenure'], bins=range(1, 80, 12), right=False, labels=labels)
            df.drop('tenure', axis=1, inplace=True)

            # One-hot encoding
            categorical_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
            ]
            dummy_df = pd.get_dummies(df[categorical_cols])
            
            # Ensure all expected columns are present
            expected_columns = model.feature_names_in_
            for col in expected_columns:
                if col not in dummy_df.columns:
                    dummy_df[col] = 0
            dummy_df = dummy_df[expected_columns]

            # Get predictions and probabilities
            predictions = model.predict(dummy_df)
            probabilities = model.predict_proba(dummy_df)

            # Prepare results
            results = []
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                cust_id = df.iloc[idx].get("customerID", f"Row {idx+1}")
                
                results.append({
                    "customer_id": cust_id,
                    "prediction": "Churn" if pred == 1 else "No Churn",
                    "probability": round(prob[1] * 100, 2),  # Churn probability
                    "no_churn_prob": round(prob[0] * 100, 2),  # No churn probability
                    "probability_width": min(100, max(5, round(prob[1] * 100)))  # For progress bar (min 5% width)
                })

            return render_template("batch_input.html", results=results)

        except Exception as e:
            app.logger.error(f"Error processing batch file: {str(e)}")
            return render_template("batch_input.html", 
                                output=[f"Error processing file: {str(e)}"])

    return render_template("batch_input.html")




if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, threaded=True)