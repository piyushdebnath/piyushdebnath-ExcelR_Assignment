# Titanic Survival Prediction App using Streamlit

# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Defining the feature input interface
# This function creates a sidebar for users to input passenger features.
def get_user_input():
    st.sidebar.header('Passenger Features')
    
    # Feature 1: Ticket Class (Pclass)
    Pclass = st.sidebar.selectbox('Ticket Class (1 = First, 2 = Second, 3 = Third)', [1, 2, 3])
    
    # Feature 2: Sex (encoded as 0 for Female and 1 for Male)
    Sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    
    # Feature 3: Age (Slider for selecting passenger age)
    Age = st.sidebar.slider('Age', 0, 80, 30)
    
    # Feature 4: Number of Siblings/Spouses Aboard (SibSp)
    SibSp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
    
    # Feature 5: Number of Parents/Children Aboard (Parch)
    Parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
    
    # Feature 6: Fare Paid (Fare)
    Fare = st.sidebar.slider('Fare Paid', 0.0, 512.0, 32.2)
    
    # Feature 7: Embarked at Queenstown (Embarked_Q, encoded as 0 or 1)
    Embarked_Q = st.sidebar.selectbox('Embarked at Queenstown? (0 = No, 1 = Yes)', [0, 1])
    
    # Feature 8: Embarked at Southampton (Embarked_S, encoded as 0 or 1)
    Embarked_S = st.sidebar.selectbox('Embarked at Southampton? (0 = No, 1 = Yes)', [0, 1])

    # Combine features into an array
    features = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]).reshape(1, -1)
    return features

# Streamlit App Layout
# Explanation: The layout includes a title, description, and user input for prediction.
st.title('Titanic Survival Prediction App')
st.markdown(
    """
    This app predicts the survival probability of a passenger on the Titanic
    based on their features.
    """
)

# Get user input
user_features = get_user_input()

# Predict survival probability
# Explanation: When the 'Predict Survival' button is clicked, the model predicts the survival probability based on user inputs.
if st.button('Predict Survival'):
    # Predict the survival probability
    survival_probability = model.predict_proba(user_features)[0][1]

    # Convert probability to a binary prediction
    prediction = 'Survived' if survival_probability >= 0.5 else 'Not Survived'

    # Display prediction results
    st.subheader('Prediction Results:')
    st.write(f'Prediction: {prediction}')
    st.write(f'Survival Probability: {survival_probability:.2f}')

# Instructions for deployment:
# Explanation: Steps to run the app locally or deploy it online are provided.
st.markdown(
    "Titanic Survival Prediction App using Streamlit "
)
