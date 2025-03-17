# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 02:02:06 2025

@author: nibir
"""

#main file to deploy

import numpy as np
import pickle
import streamlit as st
import os

# Title of the app
#st.title("File Directory Checker")
# Get the current working directory
#current_directory = os.getcwd()

# Display the current directory
#st.write("Current Directory:", current_directory)

# List files and directories in the current directory
#files_and_dirs = os.listdir(current_directory)

# Display the files and directories
#st.write("Files and Directories:")
#for item in files_and_dirs:
#    st.write(item)


#model_path = os.path.join(os.path.dirname(__file__), "artifacts", "DT_model.sav")
# Debugging: Check if the path is correct
#st.write("Model Path:", model_path)
#st.write("File Exists:", os.path.exists(model_path))

# 1. Decision tree
dt_model = pickle.load(open("artifacts/DT_model.sav", 'rb'))
#model_path = os.path.join(os.path.dirname(__file__), "artifacts", "DT_model.sav")
#with open(model_path, 'rb') as model_file:
#    dt_model = pickle.load(model_file)


# 2. KNN
knn_model = pickle.load(open("artifacts/knn_model.sav", 'rb'))
#modelknn_path = os.path.join(os.path.dirname(__file__), "artifacts", "knn_model.sav")
#with open(modelknn_path, 'rb') as modelknn_file:
#    knn_model = pickle.load(modelknn_file)


# 3. NBayes
nb_model = pickle.load(open("artifacts/NBayes.sav", 'rb'))
#modelnb_path = os.path.join(os.path.dirname(__file__), "artifacts", "NBayes.sav")
#with open(modelnb_path, 'rb') as modelnb_file:
#    nb_model = pickle.load(modelnb_file)


# 4. Random Forest
rf_model = pickle.load(open("artifacts/Rforest.sav", 'rb'))
#modelrf_path = os.path.join(os.path.dirname(__file__), "artifacts", "Rforest.sav")
#with open(modelrf_path, 'rb') as modelrf_file:
#    rf_model = pickle.load(modelrf_file)


# 5. SVM
svm_model = pickle.load(open("artifacts/svm_model.sav", 'rb'))
#modelsvm_path = os.path.join(os.path.dirname(__file__), "artifacts", "svm_model.pkl")
#with open(modelsvm_path, 'rb') as modelsvm_file:
#    svm_model = pickle.load(modelsvm_file)
    
#load_model = pickle.load(open("/artifacts/DT_model.sav", 'rb'))

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

disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

#predict function for Decision Tree
def predict_dt(clf, symptoms):
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
    
#predict function for KNN 
def predict_knn(clf, symptoms):
    l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    input_test = np.array(l2).reshape(1, -1)
    predicted = clf.predict(input_test)[0]
    return predicted if isinstance(predicted, str) else "Error: Unexpected prediction output"

#predict function for Naive Bayes 
def predict_nb(clf, symptoms):
    l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    input_test = np.array(l2).reshape(1, -1)  # Reshape to match training data
    predicted = clf.predict(input_test)[0]  # This is already the disease name
    return predicted

#predict function for Random Forest
def predict_rf(clf, symptoms):
    l2 = [1 if symptom in l1 else 0 for symptom in l1]
    input_test = np.array(l2).reshape(1, -1)  # Reshape to match training data
    predicted = clf.predict(input_test)[0]

    if isinstance(predicted, str):  # If prediction is a string, convert it to index
        predicted = disease.index(predicted) if predicted in disease else -1

    return disease[predicted] if 0 <= predicted < len(disease) else "Not Found"


#predict function for Support Vector Machine (SVM)
def predict_svm(clf, symptoms):
    l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    input_test = np.array(l2).reshape(1, -1)
    predicted = clf.predict(input_test)[0]
    return predicted if isinstance(predicted, str) else "Error: Unexpected prediction output"


def main():
    #title for webpage
    st.title('Human Disease Detection')
    
    #getting the user data
    # Mandatory input for the first two symptoms
    symptoms = []
    for i in range(1, 3):  # Loop for the first two mandatory symptoms
        symptom = st.selectbox(f"Enter {i} symptom (mandatory): ",options=[''] + l1)
        if symptom:  # Only append if the user has entered a symptom
            symptoms.append(symptom)

    # Optional input for the next three symptoms
    for i in range(3, 6):  # Loop for the next three optional symptoms
        symptom = st.selectbox(f"Enter {i} symptom (optional, press Enter to skip): ", options= [''] + l1)
        if symptom:  # Only append if the user has entered a symptom
            symptoms.append(symptom)
    # Predicting the disease
    #predicted_disease = ' '
    
    #button for prediction
    if st.button("Predict"):
        st.write(f"The symptoms you have enterd: {symptoms}")
        if len(symptoms) < 2:
            st.error("Please select at least two mandatory symptoms.")
        else:
             predicted_dt = predict_dt(dt_model, symptoms)
             st.success(f"Decision Tree Predicted: {predicted_dt}")
             predicted_knn = predict_knn(knn_model, symptoms)
             st.success(f"KNN model Predicted: {predicted_knn}")
             predicted_nb = predict_dt(nb_model, symptoms)
             st.success(f"Naive Bayes Predicted: {predicted_nb}")
             predicted_rf = predict_dt(rf_model, symptoms)
             st.success(f"Random Forest Predicted: {predicted_rf}")
             predicted_svm = predict_dt(svm_model, symptoms)
             st.success(f"SVM model Predicted: {predicted_svm}")
        
    
    

if __name__ == '__main__':
    main()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
