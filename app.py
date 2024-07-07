from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pyttsx3

app = Flask(__name__)

# Load dataset
df = pd.read_csv('diabetes.csv')

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_gender = request.form.get('gender')
        if selected_gender == 'male':
            return redirect(url_for('male.html'))
        elif selected_gender == 'female':
            return redirect(url_for('female.html'))
    return render_template('index.html')

@app.route('/female', methods=['GET', 'POST'])
def female():
    if request.method == 'POST':
        # Get user data from form
        user_data = {
            'pregnancies': int(request.form['pregnancies']),
            'glucose': int(request.form['glucose']),
            'bp': int(request.form['bp']),
            'skinthickness': int(request.form['skinthickness']),
            'insulin': int(request.form['insulin']),
            'bmi': float(request.form['bmi']),
            'dpf': float(request.form['dpf']),
            'age': int(request.form['age'])
        }
        user_data_df = pd.DataFrame(user_data, index=[0])

        # Prediction and probability
        user_result = rf.predict(user_data_df)
        user_prob = rf.predict_proba(user_data_df)
        confidence = user_prob[0][user_result[0]]

        result = 'Diabetic' if user_result[0] == 1 else 'Not Diabetic'

        # Visualization
        color = 'blue' if user_result[0] == 0 else 'red'
        plots = create_plots(user_data_df, color)

        # Precautions and Advice
        if user_result[0] == 0:
            precautions = precautions_low_risk
            advice = advice_low_risk
        else:
            precautions = precautions_high_risk
            advice = advice_high_risk

        specific_advice = []
        if user_data['glucose'] > 140:
            specific_advice.append("Monitor blood glucose levels more frequently.")
            specific_advice.append("Reduce intake of sugary foods and beverages.")
            specific_advice.append("Consult a dietitian for a personalized meal plan.")
            specific_advice.append("Eat meals at regular intervals to maintain stable blood sugar levels.")

        if user_data['bp'] > 120:
            specific_advice.append("Limit sodium intake like salt, cheese, cereals as it contains high sodium.")
            specific_advice.append("Incorporate more potassium-rich foods like leafy vegetables, bananas, etc. in your diet.")
            specific_advice.append("Maintain a healthy weight and engage in regular physical activity.")

        return render_template('result.html', result=result, confidence=confidence, plots=plots, user_data=user_data, precautions=precautions, advice=advice, specific_advice=specific_advice)
    return render_template('female.html')

@app.route('/male', methods=['GET', 'POST'])
def male():
    if request.method == 'POST':
        # Get user data from form
        user_data = {
            'glucose': int(request.form['glucose']),
            'bp': int(request.form['bp']),
            'skinthickness': int(request.form['skinthickness']),
            'insulin': int(request.form['insulin']),
            'bmi': float(request.form['bmi']),
            'dpf': float(request.form['dpf']),
            'age': int(request.form['age'])
        }
        user_data_df = pd.DataFrame(user_data, index=[1])

        # Prediction and probability
        user_result = rf.predict(user_data_df)
        user_prob = rf.predict_proba(user_data_df)
        confidence = user_prob[0][user_result[0]]

        result = 'Diabetic' if user_result[0] == 1 else 'Not Diabetic'

        # Visualization
        color = 'blue' if user_result[0] == 0 else 'red'
        plots = create_plots(user_data_df, color)

        # Precautions and Advice
        if user_result[0] == 0:
            precautions = precautions_low_risk
            advice = advice_low_risk
        else:
            precautions = precautions_high_risk
            advice = advice_high_risk

        specific_advice = []
        if user_data['glucose'] > 140:
            specific_advice.append("Monitor blood glucose levels more frequently.")
            specific_advice.append("Reduce intake of sugary foods and beverages.")
            specific_advice.append("Consult a dietitian for a personalized meal plan.")
            specific_advice.append("Eat meals at regular intervals to maintain stable blood sugar levels.")

        if user_data['bp'] > 120:
            specific_advice.append("Limit sodium intake like salt, cheese, cereals as it contains high sodium.")
            specific_advice.append("Incorporate more potassium-rich foods like leafy vegetables, bananas, etc. in your diet.")
            specific_advice.append("Maintain a healthy weight and engage in regular physical activity.")

        return render_template('result.html', result=result, confidence=confidence, plots=plots, user_data=user_data, precautions=precautions, advice=advice, specific_advice=specific_advice)
    return render_template('male.html')


@app.route('/graphs')
def graphs():
    plots = os.listdir('static')
    return render_template('graphs.html', plots=plots)

def create_plots(user_data, color):
    plots = []

    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Age vs Pregnancies
    fig_preg = plt.figure()
    sns.scatterplot(x='age', y='pregnancies', data=df, hue='Outcome', palette='Greens')
    sns.scatterplot(x=user_data['age'], y=user_data['pregnancies'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_preg_path = os.path.join('static', 'fig_preg.png')
    fig_preg.savefig(fig_preg_path)
    plt.close(fig_preg)
    plots.append('fig_preg.png')

    # Age vs Glucose
    fig_glucose = plt.figure()
    sns.scatterplot(x='age', y='glucose', data=df, hue='Outcome', palette='magma')
    sns.scatterplot(x=user_data['age'], y=user_data['glucose'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_glucose_path = os.path.join('static', 'fig_glucose.png')
    fig_glucose.savefig(fig_glucose_path)
    plt.close(fig_glucose)
    plots.append('fig_glucose.png')

    # Age vs Bp
    fig_bp = plt.figure()
    sns.scatterplot(x='age', y='bp', data=df, hue='Outcome', palette='Reds')
    sns.scatterplot(x=user_data['age'], y=user_data['bp'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_bp_path = os.path.join('static', 'fig_bp.png')
    fig_bp.savefig(fig_bp_path)
    plt.close(fig_bp)
    plots.append('fig_bp.png')

    # Age vs Skin Thickness
    fig_st = plt.figure()
    sns.scatterplot(x='age', y='skinthickness', data=df, hue='Outcome', palette='Blues')
    sns.scatterplot(x=user_data['age'], y=user_data['skinthickness'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_st_path = os.path.join('static', 'fig_st.png')
    fig_st.savefig(fig_st_path)
    plt.close(fig_st)
    plots.append('fig_st.png')

    # Age vs Insulin
    fig_i = plt.figure()
    sns.scatterplot(x='age', y='insulin', data=df, hue='Outcome', palette='rocket')
    sns.scatterplot(x=user_data['age'], y=user_data['insulin'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_i_path = os.path.join('static', 'fig_i.png')
    fig_i.savefig(fig_i_path)
    plt.close(fig_i)
    plots.append('fig_i.png')

    # Age vs BMI
    fig_bmi = plt.figure()
    sns.scatterplot(x='age', y='bmi', data=df, hue='Outcome', palette='rainbow')
    sns.scatterplot(x=user_data['age'], y=user_data['bmi'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 70, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_bmi_path = os.path.join('static', 'fig_bmi.png')
    fig_bmi.savefig(fig_bmi_path)
    plt.close(fig_bmi)
    plots.append('fig_bmi.png')

    # Age vs Dpf
    fig_dpf = plt.figure()
    sns.scatterplot(x='age', y='dpf', data=df, hue='Outcome', palette='YlOrBr')
    sns.scatterplot(x=user_data['age'], y=user_data['dpf'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 3, 0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    fig_dpf_path = os.path.join('static', 'fig_dpf.png')
    fig_dpf.savefig(fig_dpf_path)
    plt.close(fig_dpf)
    plots.append('fig_dpf.png')

    return plots

precautions_low_risk = [
    "Take adequate quantity of dietary fibers.",
    "Eating regular, balanced meals can help prevent fluctuations in blood sugar.",
    "Engage in regular physical activity like walking, swimming, or cycling.",
    "Monitor your blood sugar levels regularly.",
    "Stay hydrated by drinking plenty of water throughout the day."
]

advice_low_risk = [
    "Consult with a healthcare professional for regular check-ups and follow-up appointments.",
    "Practice stress management techniques such as meditation, yoga, deep breathing exercises, or hobbies that help you relax.",
    "Be aware of your family history as genetics can play a role in diabetes risk.",
    "Recognize that certain age groups (45 and older) have a higher risk of developing diabetes."
]

precautions_high_risk = [
    "Count carbohydrates and spread your carbohydrate intake evenly throughout the day to manage blood sugar levels.",
    "Eat a variety of foods, including vegetables, fruits, whole grains, lean proteins, and healthy fats.",
    "Count carbohydrates and spread your carbohydrate intake evenly throughout the day to manage blood sugar levels.",
    "Eat a variety of foods, including vegetables, fruits, whole grains, lean proteins, and healthy fats.",
    "Take prescribed medications as directed by your doctor.",
    "Stay physically active with exercises suitable for your health condition.",
    "Avoid sugary foods and beverages."
]

advice_high_risk = [
    "Schedule regular appointments with your healthcare provider for monitoring and management of diabetes."
    "Seek help to quit smoking and drinking as it increases the risk of diabetes complications."
    "Ensure you get 7-9 hours of quality sleep per night, as poor sleep can affect blood sugar levels."
    "Always carry fast-acting carbohydrates, such as glucose tablets or juice, to treat low blood sugar."
]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

