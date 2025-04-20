from flask import Flask, request, jsonify, make_response
import pandas as pd
import joblib
from flask_cors import CORS
from collections import OrderedDict
import json

# Load models and encoders
food_model = joblib.load("goal_classifier.pkl")
exercise_model = joblib.load("exercise_classifier.pkl")
encoders = joblib.load("encoders.pkl")
df = pd.read_csv("fitness_meal_plan_with_exercises.csv")

le_gender = encoders['gender']
le_workout = encoders['workout']
le_goal = encoders['goal']
le_exercise = encoders['exercise']
preprocessor = encoders['preprocessor']

app = Flask(__name__)
CORS(app)  # Enable for frontend communication (e.g., Flutter)

def calculate_bmi(weight_kg, height_cm):
    return weight_kg / ((height_cm / 100) ** 2)

def get_meal_plan(week, day):
    filtered = df[(df['Week'] == week) & (df['Day'] == day)]
    if not filtered.empty:
        row = filtered.iloc[0]
        return OrderedDict([
            ("Breakfast", {
                "Meal": row["Breakfast"],
                "Calories": int(row["Calories_Breakfast"])
            }),
            ("Snack_1", {
                "Meal": row["Snack_1"],
                "Calories": int(row["Calories_Snack_1"])
            }),
            ("Lunch", {
                "Meal": row["Lunch"],
                "Calories": int(row["Calories_Lunch"])
            }),
            ("Snack_2", {
                "Meal": row["Snack_2"],
                "Calories": int(row["Calories_Snack_2"])
            }),
            ("Dinner", {
                "Meal": row["Dinner"],
                "Calories": int(row["Calories_Dinner"])
            }),
        ])
    return OrderedDict()

def get_exercise(week, day):
    filtered = df[(df['Week'] == week) & (df['Day'] == day)]
    if not filtered.empty:
        row = filtered.iloc[0]
        try:
            row['Exercise_Name'] = le_exercise.inverse_transform([row['Exercise_Name']])[0]
        except:
            pass  # Already a string
        return row[['Exercise_Name', 'Exercise_Description', 'Exercise_Duration']].to_dict()
    return {}

def format_meal_plan(meal_plan, bmi, exercise):
    meal_order = ["Breakfast", "Snack_1", "Lunch", "Snack_2", "Dinner"]
    sorted_meals = OrderedDict((meal, meal_plan[meal]) for meal in meal_order if meal in meal_plan)
    total_calories = sum(meal["Calories"] for meal in sorted_meals.values())

    return {
        "BMI": bmi,
        "Exercise": exercise,
        "Meal_Plan": sorted_meals,
        "Total_Calories": total_calories
    }

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    try:
        user_input = {
            'Gender': le_gender.transform([data['Gender']])[0],
            'Age': data['Age'],
            'Height_cm': data['Height_cm'],
            'Weight_kg': data['Weight_kg'],
            'Workout_History': le_workout.transform([data['Workout_History']])[0],
            'Goal': le_goal.transform([data['Goal']])[0],
            'Week': data['Week'],
            'Day': data['Day']
        }

        user_input['BMI'] = calculate_bmi(user_input['Weight_kg'], user_input['Height_cm'])
        user_df = pd.DataFrame([user_input])
        user_X = preprocessor.transform(user_df)

        # Predictions not returned to frontend, still useful if needed internally
        food_model.predict(user_X)
        exercise_model.predict(user_X)

        bmi = user_input['BMI']

        if data['choice'] == 'meal':
            meal_plan = get_meal_plan(user_input['Week'], user_input['Day'])
            response_data = {
                "BMI": bmi,
                "Meal_Plan": meal_plan,
                "Total_Calories": sum(meal["Calories"] for meal in meal_plan.values())
            }

        elif data['choice'] == 'exercise':
            exercise = get_exercise(user_input['Week'], user_input['Day'])
            response_data = {
                "BMI": bmi,
                "Exercise": exercise
            }

        else:
            return jsonify({"error": "Invalid choice. Must be 'meal' or 'exercise'"}), 400

        return make_response(json.dumps(response_data, ensure_ascii=False), 200, {'Content-Type': 'application/json'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

