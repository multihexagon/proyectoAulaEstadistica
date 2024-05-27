from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Listas para almacenar los datos
data_x = []
data_y = []

@app.context_processor
def utility_processor():
    return dict(zip=zip)

@app.route('/')
def index():
    return render_template('index.html', data_x=data_x, data_y=data_y)

@app.route('/add', methods=['POST'])
def add():
    try:
        x = float(request.form['x'])
        y = float(request.form['y'])
        data_x.append([x])
        data_y.append(y)
    except ValueError:
        pass  # Ignorar valores no válidos
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convertir listas a arrays numpy
        X = np.array(data_x)
        y = np.array(data_y)

        # Crear y entrenar el modelo
        model = LinearRegression()
        model.fit(X, y)

        # Obtener el valor de x para predecir
        x_new = float(request.form['x_new'])
        prediction = model.predict([[x_new]])[0]

        # Generar gráfico
        plt.figure()
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, model.predict(X), color='red', label='Regression line')
        plt.scatter([x_new], [prediction], color='green', label='Prediction')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Linear Regression')

        # Guardar gráfico en memoria
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', data_x=data_x, data_y=data_y, prediction=prediction, x_new=x_new, graph_url=graph_url)
    except ValueError:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
