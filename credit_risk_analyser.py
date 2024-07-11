import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import joblib
import pickle

# Function to load the model
@st.cache_resource
def load_model(path):
    loaded_model = joblib.load(path)
    return loaded_model


# Function to make prediction
def make_prediction(data):
    model_path = 'Logistic_regression_model.pkl'
    model = load_model(model_path)
    with open('Random_Forest_model.pkl', 'rb') as model_file_rf:
        loaded_model_r_f = pickle.load(model_file_rf)
    # Extract the features from the form data
    features = {
        'person_home_ownership_MORTGAGE': 0,
        'person_home_ownership_OTHER': 0,
        'person_home_ownership_OWN': 0,
        'person_home_ownership_RENT': 0,
        'loan_intent_DEBTCONSOLIDATION': 0,
        'loan_intent_EDUCATION': 0,
        'loan_intent_HOMEIMPROVEMENT': 0,
        'loan_intent_MEDICAL': 0,
        'loan_intent_PERSONAL': 0,
        'loan_intent_VENTURE': 0,
        'loan_grade_A': 0,
        'loan_grade_B': 0,
        'loan_grade_C': 0,
        'loan_grade_D': 0,
        'loan_grade_E': 0,
        'loan_grade_F': 0,
        'loan_grade_G': 0,
        'cb_person_default_on_file_N': 0,
        'cb_person_default_on_file_Y': 0,
        'person_age': data['person_age'],
        'person_income': data['person_income'],
        'person_emp_length': data['person_emp_length'],
        'loan_amnt': data['loan_amnt'],
        'loan_int_rate': data['loan_int_rate'],
        "loan_percent_income": data["loan_amnt"] / data["person_income"],
        'cb_person_cred_hist_length': data['cb_person_cred_hist_length']
    }
    
    # Update binary features based on input data
    features[f'person_home_ownership_{data["person_home_ownership"]}'] = 1
    features[f'loan_intent_{data["loan_intent"]}'] = 1
    features[f'loan_grade_{data["loan_grade"]}'] = 1
    features[f'cb_person_default_on_file_{data["cb_person_default_on_file"]}'] = 1

    input_features = pd.DataFrame([features])
    with open('scaler.pkl', 'rb') as file:
        sfile=pickle.load(file)
    features_to_normalize=["person_age",	"person_income",	"person_emp_length",	"loan_amnt",	"cb_person_cred_hist_length"]

    input_features[features_to_normalize]=sfile.transform(input_features[features_to_normalize])
    # Make prediction
    prediction = loaded_model_r_f.predict_proba(input_features)[0, 1]  # Get the probability of the positive class
    
    return prediction

# Initialize session state to store gauge value
if 'prediction_value' not in st.session_state:
    st.session_state.prediction_value = 0.0

# Streamlit layout setup
st.title('Credit Risk Analyzer')
st.markdown('### Enter your details below:')


# Sidebar for gauge
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Credit Worthiness</h1>", unsafe_allow_html=True)
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#F65164;" /> """, unsafe_allow_html=True)
    gauge_placeholder = st.empty()

# Define columns for layout: 2 columns for input fields
col1, col2 = st.columns([1, 1])

# Input fields in the first column

with st.form(key='input_form'):
    with col1:
        person_age = st.number_input('Age', step=1)        
        loan_intent = ((st.selectbox('Loan Intent', ['-Select-', 'Debt Consolidation', 'Education', 'Home Improvement', 'Medical', 'Personal', 'Venture'], index=0)).upper()).replace(" ","")
        person_home_ownership = ((st.selectbox('Home Ownership', ['-Select-', 'Mortgage', 'Other', 'Own', 'Rent'], index=0)).upper()).replace(" ","")
        loan_grade = ((st.selectbox('Loan Grade', ['-Select-', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0))).replace(" ","")
        cb_person_default_on_file = ((st.selectbox('Credit Bureau Default on File', ['-Select-', 'No', 'Yes'], index=0)).upper()).replace(" ","")

    with col2:
        loan_amnt = st.number_input('Loan Amount ($, Yearly)')
        person_income = st.number_input('Income ($, Yearly)')
        loan_int_rate = st.number_input('Loan Interest Rate (%)', step=0.01)
        person_emp_length = st.number_input('Employment Length (years)', step=1)
        cb_person_cred_hist_length = st.number_input('Credit History Length (years)', step=1)

        # Predict button
    submit_button = st.form_submit_button(label='Predict')
        
    if submit_button:
        if (person_home_ownership == '-SELECT-' or loan_intent == '-SELECT-' or loan_grade == '-SELECT-' or cb_person_default_on_file == '-SELECT-'):
            st.error('Please select valid options for all fields.')
        if (person_age<15):
            st.error("The Borrower can't be aged below 15!")
        if (person_income==0):
            st.error("The income amount can't be 0")
        else:
            input_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_emp_length': person_emp_length,
                'loan_amnt': loan_amnt,
                'loan_int_rate': loan_int_rate,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
                'person_home_ownership': person_home_ownership,
                'loan_intent': loan_intent,
                'loan_grade': loan_grade,
                'cb_person_default_on_file': cb_person_default_on_file[0]
            }
            # Make prediction and display
            try:
                prediction = make_prediction(input_data)
                st.session_state.prediction_value = prediction  # Update session state
                st.success(f'Loan Approval Probability: {prediction:.2%}')
                
                # Gauge Visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={'text': "Approval Probability"},
                    gauge={'axis': {'range': [0, 100]},
                            'bar': {'color': 'black'},
                            'steps': [
                                {'range': [0, 20], 'color': 'red'},
                                {'range': [20, 40], 'color': 'orange'},
                                {'range': [40, 60], 'color': 'yellow'},
                                {'range': [60, 80], 'color': 'lightgreen'},
                                {'range': [80, 100], 'color': 'green'}
                            ],
                            'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 1, 'value': 65}},
                    number={'suffix': '%'}

                
                ))
               
                gauge_placeholder.plotly_chart(fig)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Initial gauge visualization in the sidebar
if st.session_state.prediction_value == 0.0:
    initial_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0.0,
        title={'text': "Approval Probability"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': 'green'},
               'steps': [
                   {'range': [0, 20], 'color': 'red'},
                   {'range': [20, 40], 'color': 'orange'},
                   {'range': [40, 60], 'color': 'yellow'},
                   {'range': [60, 80], 'color': 'lightgreen'},
                   {'range': [80, 100], 'color': 'green'}],
               'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 1, 'value': 65}},
        number={'suffix': '%'}
    ))
    gauge_placeholder.plotly_chart(initial_fig)

