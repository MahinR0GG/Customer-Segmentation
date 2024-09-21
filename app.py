from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('plot', filename=file.filename))
    except Exception as e:
        return f"File upload error: {e}", 500
    return redirect(request.url)

@app.route('/results')
def results():
    try:
        filename = request.args.get('filename')
        algorithm = request.args.get('algorithm')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)

        required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
        if not all(col in df.columns for col in required_columns):
            return f"Error: Dataset is missing required columns: {', '.join(required_columns)}"

        df = df[required_columns]
        X = df.values

        if algorithm == 'kmeans':
            model = KMeans(n_clusters=8, random_state=42)
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters
            accuracy = model.inertia_  # KMeans inertia as a proxy for "accuracy"
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=5, min_samples=5)
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters
            accuracy = (clusters != -1).mean() * 100  # Percentage of points in non-noise clusters
        else:
            return "Invalid algorithm selected"

        fig = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Cluster', color='Cluster')
        graphJSON = fig.to_json()

        return render_template('results.html', graphJSON=graphJSON, accuracy=accuracy)

    except Exception as e:
        return f"Error processing results: {e}", 500



@app.route('/plot', methods=['GET', 'POST'])
def plot():
    try:
        filename = request.args.get('filename')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return f"Error: Dataset is missing required columns: {', '.join(missing_cols)}"

        # Generate Bar Plots
        age_income_bar = px.bar(df, x="Age", y="Annual Income (k$)", title="Age vs Annual Income")
        age_spending_bar = px.bar(df, x="Age", y="Spending Score (1-100)", title="Age vs Spending Score")
        income_spending_bar = px.bar(df, x="Annual Income (k$)", y="Spending Score (1-100)", title="Annual Income vs Spending Score")
        
        gender_distribution = df['Gender'].value_counts().reset_index()
        gender_distribution.columns = ['Gender', 'Count']
        gender_bar = px.bar(gender_distribution, x='Gender', y='Count', title='Gender Distribution')

        # Converting plots to JSON
        age_income_bar_json = age_income_bar.to_json()
        age_spending_bar_json = age_spending_bar.to_json()
        income_spending_bar_json = income_spending_bar.to_json()
        gender_bar_json = gender_bar.to_json()

        return render_template('plot.html',
                               age_income_bar_json=age_income_bar_json,
                               age_spending_bar_json=age_spending_bar_json,
                               income_spending_bar_json=income_spending_bar_json,
                               gender_bar_json=gender_bar_json,
                               filename=filename)
    except Exception as e:
        return f"Error generating plots: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
