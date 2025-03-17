import numpy as np
import pickle

load_model = pickle.load(open("D:/Codes/MiniProject/model_repo/DT_model.sav", 'rb'))

#List of the symptoms is listed here in list l1.
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

def predict_disease(clf, symptoms):
    # Convert symptoms into a feature vector
    l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    
    # Ensure input is in the correct format for prediction
    input_test = np.array(l2).reshape(1, -1)  # Reshape to match training data

    # Predict using the trained classifier
    predicted = clf.predict(input_test)[0]

    # Ensure predicted value is handled correctly
    if isinstance(predicted, (int, np.integer)):  # If it’s an index
        return disease[predicted] if predicted < len(disease) else "Not Found"
    elif isinstance(predicted, str):  # If it directly returns the disease name
        return predicted
    else:
        return "Error: Unexpected prediction output"


# Main execution
#clf = train_decision_tree(X, y)

# Taking input from the user
symptoms = []

# Mandatory input for the first two symptoms
for i in range(1, 3):  # Loop for the first two mandatory symptoms
    symptom = input(f"Enter {i} symptom (mandatory): ")
    symptoms.append(symptom)

# Optional input for the next three symptoms
for i in range(3, 6):  # Loop for the next three optional symptoms
    symptom = input(f"Enter {i} symptom (optional, press Enter to skip): ")
    if symptom:  # Only append if the user has entered a symptom
        symptoms.append(symptom)

print(symptoms)
print(type(symptoms))
# Predicting the disease
predicted_disease = predict_disease(load_model, symptoms)
print(f"The predicted disease is: {predicted_disease}")