from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


app = Flask(__name__)


@app.route('/')
@app.route('/home', methods=['POST', 'GET'])
def home():
    house_price = pd.read_csv('D:/python/Machine Learning/housing.csv')
    house_price.dropna(inplace=True)

    def remove_outliers(dataset, columns):
        q3 = dataset[columns].quantile(0.75)
        q1 = dataset[columns].quantile(0.25)
        iqr = q3 - q1
        upper_fence = q3 + 1.5*iqr
        dataset.loc[dataset[columns] >= upper_fence, columns] = upper_fence
    remove_outliers(house_price, 'total_rooms')
    remove_outliers(house_price, 'total_bedrooms')
    remove_outliers(house_price, 'population')
    remove_outliers(house_price, 'households')
    remove_outliers(house_price, 'median_income')
    remove_outliers(house_price, 'median_house_value')
    x = house_price[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                     'total_bedrooms', 'population', 'households', 'median_income']]
    y = house_price[['median_house_value']]
    scale = StandardScaler()
    x = scale.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    gradient = GradientBoostingRegressor(
        learning_rate=0.04, max_depth=10, subsample=0.5)
    gradient = gradient.fit(x_train, y_train)
    predict = gradient.predict(x_test)
    longitude = ""
    latitude = ""
    housing_median_age = ""
    total_rooms = ""
    total_bedrooms = ""
    population = ""
    households = ""
    median_income = ""
    sample_data = ''
    result_prediction = ''
    if request.method == 'POST':
        longitude = request.form['longitude']
        latitude = request.form['latitude']
        housing_median_age = request.form['housing_median_age']
        total_rooms = request.form['total_rooms']
        total_bedrooms = request.form['total_bedrooms']
        population = request.form['population']
        households = request.form['households']
        median_income = request.form['median_income']
        sample_data = [latitude, longitude, housing_median_age, total_rooms,
                       total_bedrooms, population, households, median_income]
        data = np.array(sample_data).reshape(1, -1)
        result_prediction = int(gradient.predict(data))
        print(int(result_prediction))

    return render_template('home.html',
                           longitude=longitude, latitude=latitude, housing_median_age=housing_median_age, total_rooms=total_rooms, total_bedrooms=total_bedrooms, population=population, households=households, median_income=median_income, result_prediction=result_prediction)


if __name__ == '__main__':
    app.run(debug=True)
