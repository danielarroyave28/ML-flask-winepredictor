from flask import Flask, render_template, url_for, jsonify, redirect, request
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from flask_bootstrap import Bootstrap5
from custom_transformer import CombinedAttributesAdder
import joblib
import numpy as np
import json

app = Flask(__name__)
Bootstrap5(app)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
le = joblib.load('labelEncoder.joblib')
model = joblib.load('final_prod_model.pkl')


class WineForm(FlaskForm):
    fixed_acidity = FloatField("fixed acidity", validators=[DataRequired(), NumberRange(min=4.0, max=20.0)],
                               default=4.0)
    volatile_acidity = FloatField("volatile acidity", validators=[DataRequired(), NumberRange(min=0.1, max=2.0)])
    citric_acid = FloatField("citric acid", validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    residual_sugar = FloatField("residual sugar", validators=[DataRequired(), NumberRange(min=0.5, max=20.0)])
    chlorides = FloatField("chlorides", validators=[DataRequired(), NumberRange(min=0.01, max=1.0)])
    free_sulfur_dioxide = FloatField("free sulfur dioxide",
                                     validators=[DataRequired(), NumberRange(min=1.0, max=100.0)])
    total_sulfur_dioxide = FloatField("total sulfur dioxide",
                                      validators=[DataRequired(), NumberRange(min=1.0, max=300.0)])
    density = FloatField("density",
                         validators=[DataRequired(), NumberRange(min=0.9, max=1.1)])
    pH = FloatField("pH", validators=[DataRequired(), NumberRange(min=2.7, max=4.1)], default=3.0)
    sulphates = FloatField("sulphates", validators=[DataRequired(), NumberRange(min=0.3, max=2.0)])
    alcohol = FloatField("alcohol", validators=[DataRequired(), NumberRange(min=8.0, max=15.0)], default=8)
    submit = SubmitField("Predict")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    form = WineForm()
    if form.validate_on_submit():
        features = [x for x in form.data.values()]
        features_array = np.array(features[:-2]).reshape(1, -1)
        print(features)
        print(features_array)

        output = np.squeeze(le.inverse_transform(model.predict(features_array)))
        print(output)
        # print(output.shape)
        # print(output.reshape(1, -1))
        # print(output[0])
        # print(np.squeeze(output))

        return render_template("predict_wine.html", form=form,
                               prediction=f"The quality from 3 to 8 of this wine is: {output}")

    else:
        return render_template("predict_wine.html", form=form)


@app.route('/predict_api', methods=["GET", "POST"])
def predict_api():
    #data = request.args.values() # Sending data via params
    data = request.form.values() # Sending data via Body form url encoded
    features = list(data)
    print(features)

    final_features = np.array(features).astype(float).reshape(1, -1)
    print(final_features)

    output = np.squeeze(le.inverse_transform(model.predict(final_features)))

    return jsonify(response={"quality prediction": f"{output}"}), 200


if __name__ == "__main__":
    app.run(debug=True)
