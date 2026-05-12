import streamlit as st
import pickle
import pandas as pd

# Load trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Cities
cities = [
    'Hyderabad',
    'Bangalore',
    'Mumbai',
    'Indore',
    'Kolkata',
    'Delhi',
    'Chandigarh',
    'Jaipur',
    'Chennai',
    'Cape Town',
    'Port Elizabeth',
    'Durban',
    'Centurion',
    'East London',
    'Johannesburg',
    'Kimberley',
    'Bloemfontein',
    'Ahmedabad',
    'Cuttack',
    'Nagpur',
    'Dharamsala',
    'Visakhapatnam',
    'Pune',
    'Raipur',
    'Ranchi',
    'Abu Dhabi',
    'Sharjah',
    'Mohali',
    'Bengaluru'
]

# App title
st.title('🏏 IPL Win Predictor')

# Batting team
batting_team = st.selectbox(
    'Select Batting Team',
    sorted(teams)
)

# Bowling team
bowling_team = st.selectbox(
    'Select Bowling Team',
    sorted(teams)
)

# Prevent same team selection
if batting_team == bowling_team:
    st.error("Batting and Bowling teams cannot be same")

# Host city
selected_city = st.selectbox(
    'Select Host City',
    sorted(cities)
)

# Match target
target = st.number_input(
    'Target',
    min_value=1
)

# Current score
score = st.number_input(
    'Current Score',
    min_value=0
)

# Overs completed
overs = st.number_input(
    'Overs Completed',
    min_value=0.0,
    max_value=20.0,
    step=0.1
)

# Wickets out
wickets = st.number_input(
    'Wickets Out',
    min_value=0,
    max_value=10
)

# Prediction button
if st.button('Predict Probability'):

    runs_left = target - score

    balls_left = 120 - int(overs * 6)

    wickets_left = 10 - wickets

    # Avoid division by zero
    crr = score / overs if overs > 0 else 0

    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Input dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Prediction
    result = pipe.predict_proba(input_df)

    loss = result[0][0]

    win = result[0][1]

    # Display probabilities
    st.header(f"🏏 {batting_team} Win Probability: {round(win * 100)}%")

    st.header(f"🎯 {bowling_team} Win Probability: {round(loss * 100)}%")