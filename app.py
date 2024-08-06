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
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        algorithm = request.form.get('algorithm')
        return redirect(url_for('results', filename=file.filename, algorithm=algorithm))
    return redirect(request.url)

@app.route('/results')
def results():
    filename = request.args.get('filename')
    algorithm = request.args.get('algorithm')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X = df.values

    if algorithm == 'kmeans':
        model = KMeans(n_clusters=5, random_state=42)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=5, min_samples=5)
    else:
        return "Invalid algorithm selected"

    clusters = model.fit_predict(X)
    df['Cluster'] = clusters

    fig = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Cluster', color='Cluster')
    graphJSON = fig.to_json()

    return render_template('results.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
