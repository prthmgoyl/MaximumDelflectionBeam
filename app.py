from flask import Flask, request, jsonify , render_template
import tensorflow as tf
import numpy as np

# Load your model
model = tf.keras.models.load_model('94_97.h5')
print("strated")
app = Flask(__name__)

ids=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
data_array = np.array(ids)
data_array_reshaped = data_array.reshape((1,15, 1))
print(data_array_reshaped)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the request
    #data = request.get_json()

    ids = ["height", "width", "section_area", "length", "half_span", "a_d", "i", "mass", 
       "velocity", "energy", "pl", "pt", "concrete", "steel", "hoop"]

    # Extract values from request.form for each ID
    values = [float(request.form.get(id, 0)) for id in ids]

    # Convert the list of values to numpy array
    data_array = np.array(values)
    data_array_reshaped = data_array.reshape((1,15, 1))
    print(model.predict(data_array_reshaped))
    return str(model.predict(data_array_reshaped))


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
