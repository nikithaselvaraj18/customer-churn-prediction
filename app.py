from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the existing trained model and its feature names.
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read numerical customer details.
        senior_citizen = int(request.form["senior_citizen"])
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])

        # Read categorical customer details.
        customer_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": senior_citizen,
            "Partner": request.form["partner"],
            "Dependents": request.form["dependents"],
            "tenure": tenure,
            "PhoneService": request.form["phone_service"],
            "MultipleLines": request.form["multiple_lines"],
            "InternetService": request.form["internet_service"],
            "OnlineSecurity": request.form["online_security"],
            "OnlineBackup": request.form["online_backup"],
            "DeviceProtection": request.form["device_protection"],
            "TechSupport": request.form["tech_support"],
            "StreamingTV": request.form["streaming_tv"],
            "StreamingMovies": request.form["streaming_movies"],
            "Contract": request.form["contract"],
            "PaperlessBilling": request.form["paperless_billing"],
            "PaymentMethod": request.form["payment_method"],
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # Basic input validation.
        if senior_citizen not in (0, 1):
            return "Invalid senior citizen value.", 400

        if (
            tenure < 0 or tenure > 100
            or monthly_charges < 0
            or total_charges < 0
        ):
            return "Please enter valid non-negative customer values.", 400

        input_data = pd.DataFrame([customer_data])

        # Use the same category order as the training dataset.
        categorical_levels = {
            "gender": ["Female", "Male"],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": [
                "No", "No phone service", "Yes"
            ],
            "InternetService": [
                "DSL", "Fiber optic", "No"
            ],
            "OnlineSecurity": [
                "No", "No internet service", "Yes"
            ],
            "OnlineBackup": [
                "No", "No internet service", "Yes"
            ],
            "DeviceProtection": [
                "No", "No internet service", "Yes"
            ],
            "TechSupport": [
                "No", "No internet service", "Yes"
            ],
            "StreamingTV": [
                "No", "No internet service", "Yes"
            ],
            "StreamingMovies": [
                "No", "No internet service", "Yes"
            ],
            "Contract": [
                "Month-to-month", "One year", "Two year"
            ],
            "PaperlessBilling": ["No", "Yes"],
            "PaymentMethod": [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        }

        for col, levels in categorical_levels.items():
            value = input_data.loc[0, col]

            if value not in levels:
                return f"Invalid value for {col}.", 400

            input_data[col] = pd.Categorical(
                input_data[col],
                categories=levels
            )

        # Encode categorical data and match the saved model columns.
        input_data = pd.get_dummies(
            input_data,
            columns=list(categorical_levels.keys()),
            drop_first=True,
            dtype=int
        )

        input_data = input_data.reindex(
            columns=model_features,
            fill_value=0
        )

        # Predict churn and estimate its probability.
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        # Recommend an action based on the prediction.
        if prediction == 1:
            result = "Customer is likely to churn"

            if customer_data["Contract"] == "Month-to-month":
                recommendation = (
                    "Offer a suitable discount or incentive to "
                    "encourage a long-term contract."
                )
            elif customer_data["TechSupport"] == "No":
                recommendation = (
                    "Offer technical support assistance to "
                    "improve the customer experience."
                )
            elif tenure < 12:
                recommendation = (
                    "Offer a suitable new-customer retention "
                    "benefit and follow up regularly."
                )
            else:
                recommendation = (
                    "Offer a suitable loyalty benefit and "
                    "monitor customer satisfaction."
                )
        else:
            result = "Customer is unlikely to churn"
            recommendation = (
                "Continue regular service, customer engagement, "
                "and satisfaction monitoring."
            )

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport"
                  content="width=device-width, initial-scale=1.0">
            <title>Churn Prediction Result</title>
            <style>
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background: #f3f6fc;
                    color: #172554;
                }}
                .container {{
                    max-width: 650px;
                    margin: 35px auto;
                }}
                .header {{
                    background: #172554;
                    color: white;
                    padding: 24px;
                    border-radius: 14px 14px 0 0;
                }}
                .header h1 {{ margin-top: 0; }}
                .header p {{ color: #dbeafe; }}
                .card {{
                    background: white;
                    padding: 25px;
                    border-radius: 0 0 14px 14px;
                    box-shadow: 0 8px 28px rgba(15,23,42,.08);
                }}
                .status {{
                    padding: 16px;
                    background: #eff6ff;
                    border-radius: 10px;
                }}
                .probability {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #2563eb;
                }}
                .action {{
                    padding: 18px;
                    background: #f8fafc;
                    border-left: 5px solid #2563eb;
                    border-radius: 8px;
                    line-height: 1.6;
                }}
                .button {{
                    display: block;
                    margin-top: 24px;
                    padding: 14px;
                    text-align: center;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 9px;
                    font-weight: bold;
                }}
                .note {{
                    color: #64748b;
                    font-size: 12px;
                    line-height: 1.5;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <main class="container">
                <header class="header">
                    <h1>Customer Churn Analysis</h1>
                    <p>Prediction and Next-Best-Action Recommendation</p>
                </header>
                <section class="card">
                    <div class="status">
                        <h2>{result}</h2>
                        <p>Estimated churn probability</p>
                        <div class="probability">
                            {probability * 100:.2f}%
                        </div>
                    </div>
                    <h2>Recommended Next-Best Action</h2>
                    <div class="action">{recommendation}</div>
                    <a class="button" href="/">
                        Predict Another Customer
                    </a>
                    <p class="note">
                        The prediction is an estimate, not a guarantee
                        of future customer behaviour. The recommendation
                        is a rule-based suggestion to support retention.
                    </p>
                </section>
            </main>
        </body>
        </html>
        """

    except (ValueError, KeyError) as error:
        return (
            "Please check that all customer fields are completed "
            "with valid values. Error: " + str(error),
            400
        )


if __name__ == "__main__":
    app.run(debug=True)