# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 02:02:06 2025

@author: nibir
"""

#main file to deploy
import numpy as np
import pickle
import streamlit as st
#import sklearn

'''
import os
# Title of the app
st.title("File Directory Checker")
# Get the current working directory
current_directory = os.getcwd()

# Display the current directory
st.write("Current Directory:", current_directory)

# List files and directories in the current directory
files_and_dirs = os.listdir(current_directory)

# Display the files and directories
st.write("Files and Directories:")
for item in files_and_dirs:
    st.write(item)
'''

#model_path = os.path.join(os.path.dirname(__file__), "artifacts", "DT_model.sav")

# Debugging: Check if the path is correct
#st.write("Model Path:", model_path)
#st.write("File Exists:", os.path.exists(model_path))

# Load the model
#with open(model_path, 'rb') as model_file:
load_model = pickle.load(open("artifacts/DT_model.sav", 'rb'))

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
    
def startt():
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
    
    return 1

def main():
    #title for webpage
    st.title('Human Disease Detection')
    
    #getting the user data
    # Mandatory input for the first two symptoms
    symptoms = []
    for i in range(1, 3):  # Loop for the first two mandatory symptoms
        symptom = st.text_input(f"Enter {i} symptom (mandatory): ")
        symptoms.append(symptom)

    # Optional input for the next three symptoms
    for i in range(3, 6):  # Loop for the next three optional symptoms
        symptom = st.text_input(f"Enter {i} symptom (optional, press Enter to skip): ")
        if symptom:  # Only append if the user has entered a symptom
            symptoms.append(symptom)
    # Predicting the disease
    predicted_disease = ' '
    
    #button for prediction
    if st.button("Predict"):
        predicted_disease = predict_disease(load_model, symptoms)
        #print(f"The predicted disease is: {predicted_disease}")
        
    st.success(predicted_disease)
    

if __name__ == '__main__':
    main()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
