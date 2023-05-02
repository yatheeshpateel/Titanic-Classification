import streamlit as st
import joblib

# Load the model
model = joblib.load('titanic_survival_prediction')

# Create the Streamlit app
def app():
    # Create a title for the app
    st.title('Titanic Survival Prediction')

    # Create form to take input from the user
    with st.form(key='input_form'):
        # Create input fields for user to enter passenger details
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.slider('Age', 0, 100, 30)
        sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
        parch = st.slider('parch', 0, 10, 0)
        fare = st.slider('Fare', 0, 100, 10)
        embarked=st.selectbox('embarked', [0,1,2])

        # Create a button to submit the form
        submitted = st.form_submit_button('Predict Survival')

    # Use the model to predict survival based on user input
    if submitted:
        # Convert user input into a numpy array
        input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]
        input_data[0][1] = 1 if input_data[0][1] == 'female' else 0

        # Make the prediction
        prediction = model.predict(input_data)[0]

        # Display the prediction
        if prediction == 1:
            st.write('This passenger is predicted to have survived.')
        else:
            st.write('This passenger is predicted to have perished.')

# Run the app
if __name__ == '__main__':
    app()
