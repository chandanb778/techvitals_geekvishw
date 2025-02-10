import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import get_credentials
from fitness_data import FitnessData
from google.auth.exceptions import RefreshError
import os
import time


##################################################################################################################


# Configurations from original files (Gemini API, etc.)
genai.configure(api_key='AIzaSyDhaPCDfelx4FE_W2h9cTPmmGQZ9zwWeb0')
model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="Health & Fitness Tracker", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv("health_data2.csv")
    df['Timestamp (IST)'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp (IST)'])
    
    # Process sleep stages - convert 5-minute intervals to hours
    sleep_summary = process_sleep_stages(df)
    df = pd.merge(df, sleep_summary, on='Date', how='left')
    
    return df

def process_sleep_stages(df):
    """Process sleep stages and convert to hours"""
    # Group by date and sleep stage, count 5-minute intervals
    sleep_stages = df.groupby(['Date', 'Sleep Stage']).size().reset_index(name='count')
    
    # Convert counts (5-minute intervals) to hours
    sleep_stages['hours'] = sleep_stages['count'] * 5 / 60
    
    # Pivot to get separate columns for each sleep stage
    sleep_summary = sleep_stages.pivot(index='Date', 
                                     columns='Sleep Stage', 
                                     values='hours').reset_index()
    
    # Rename columns to match expected format
    sleep_summary = sleep_summary.rename(columns={
        'REM': 'REM Sleep',
        'Deep': 'Deep Sleep',
        'Light': 'Light Sleep'
    })
    for col in ['REM Sleep', 'Deep Sleep', 'Light Sleep']:
        if col not in sleep_summary.columns:
            sleep_summary[col] = 0
    
    return sleep_summary


# Function to calculate health metrics
def calculate_health_metrics(data):
    metrics = {
        'avg_steps': data['Total Day Steps'].mean(),
        'avg_hr': data['HR'].mean(),
        'avg_spo2': data['SpO2 (%)'].mean(),
        'avg_systolic': data['BP Systolic'].mean(),
        'avg_diastolic': data['BP Diastolic'].mean(),
        'activity_distribution': data['Activity Level'].value_counts().to_dict()
    }
    
    # Calculate sleep metrics
    sleep_columns = ['REM Sleep', 'Deep Sleep', 'Light Sleep']
    if all(col in data.columns for col in sleep_columns):
        metrics.update({
            'rem_sleep': data['REM Sleep'].iloc[0] if len(data) > 0 else 0,
            'deep_sleep': data['Deep Sleep'].iloc[0] if len(data) > 0 else 0,
            'light_sleep': data['Light Sleep'].iloc[0] if len(data) > 0 else 0
        })
    
    return metrics

def show_sleep_analysis(daily_data):
    """Display sleep analysis visualizations"""
    st.subheader("Sleep Analysis")
    
    # Count time spent in each sleep stage
    sleep_stages = daily_data['Sleep Stage'].value_counts()
    
    # Convert 5-minute intervals to hours
    sleep_hours = (sleep_stages * 5 / 60).round(2)
    
    # Create sleep distribution chart
    fig_sleep = px.pie(
        values=sleep_hours.values,
        names=sleep_hours.index,
        title='Sleep Stage Distribution (Hours)'
    )
    st.plotly_chart(fig_sleep, use_container_width=True)
    
    # Show sleep metrics
    cols = st.columns(len(sleep_hours))
    for i, (stage, hours) in enumerate(sleep_hours.items()):
        with cols[i]:
            st.metric(f"{stage}", f"{hours:.1f} hours")


# Function to generate personalized recommendations using Gemini
def get_recommendations(current_metrics, average_metrics):
    # Base metrics for the prompt
    prompt = f"""
    As a health expert, analyze these health metrics and provide specific, personalized recommendations:
    
    Current Day Metrics:
    - Steps: {current_metrics['avg_steps']:.0f}
    - Heart Rate: {current_metrics['avg_hr']:.1f} bpm
    - SpO2: {current_metrics['avg_spo2']:.1f}%
    - Blood Pressure: {current_metrics['avg_systolic']:.1f}/{current_metrics['avg_diastolic']:.1f}
    """
    
    # Add sleep metrics to prompt if available
    if all(key in current_metrics for key in ['rem_sleep', 'deep_sleep', 'light_sleep']):
        prompt += f"""
    - Sleep Distribution:
      * REM Sleep: {current_metrics['rem_sleep']:.1f} hours
      * Deep Sleep: {current_metrics['deep_sleep']:.1f} hours
      * Light Sleep: {current_metrics['light_sleep']:.1f} hours
        """
    
    # Add average metrics
    prompt += f"""
    30-Day Average Metrics:
    - Steps: {average_metrics['avg_steps']:.0f}
    - Heart Rate: {average_metrics['avg_hr']:.1f} bpm
    - SpO2: {average_metrics['avg_spo2']:.1f}%
    - Blood Pressure: {average_metrics['avg_systolic']:.1f}/{average_metrics['avg_diastolic']:.1f}
    """
    
    if all(key in average_metrics for key in ['rem_sleep', 'deep_sleep', 'light_sleep']):
        prompt += f"""
    - Sleep Distribution:
      * REM Sleep: {average_metrics['rem_sleep']:.1f} hours
      * Deep Sleep: {average_metrics['deep_sleep']:.1f} hours
      * Light Sleep: {average_metrics['light_sleep']:.1f} hours
        """
    
    prompt += """
    Provide specific recommendations in these categories:
    1. Physical Activity
    2. Cardiovascular Health
    3. Overall Wellness
    """
    
    # Add sleep category if sleep data is available
    if all(key in current_metrics for key in ['rem_sleep', 'deep_sleep', 'light_sleep']):
        prompt += """
    4. Sleep Quality
        """
    
    prompt += """
    Format as bullet points and keep it concise.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate recommendations: {str(e)}"

def show_dashboard():
    st.title("Health & Fitness Analytics Dashboard")
    
    try:
        # Load data
        df = load_data()
        
        # Sidebar for date filtering
        st.sidebar.header("Filters")
        dates = df['Date'].unique()
        selected_date = st.sidebar.selectbox("Select Date", dates)
        
        # Filter data
        daily_data = df[df['Date'] == selected_date]
        
        # Calculate metrics
        current_metrics = calculate_health_metrics(daily_data)
        average_metrics = calculate_health_metrics(df)
        
        # Create metrics comparison
        st.header("Daily vs. Average Metrics")
        cols = st.columns(4)
        
        with cols[0]:
            delta_steps = ((current_metrics['avg_steps'] - average_metrics['avg_steps'])/average_metrics['avg_steps']) * 100
            st.metric("Steps", 
                     f"{current_metrics['avg_steps']:,.0f}", 
                     f"{delta_steps:+.1f}% vs avg")
        
        with cols[1]:
            delta_hr = current_metrics['avg_hr'] - average_metrics['avg_hr']
            st.metric("Heart Rate", 
                     f"{current_metrics['avg_hr']:.1f} bpm", 
                     f"{delta_hr:+.1f} bpm vs avg")
        
        with cols[2]:
            delta_spo2 = current_metrics['avg_spo2'] - average_metrics['avg_spo2']
            st.metric("SpO2", 
                     f"{current_metrics['avg_spo2']:.1f}%", 
                     f"{delta_spo2:+.1f}% vs avg")
        
        with cols[3]:
            delta_systolic = current_metrics['avg_systolic'] - average_metrics['avg_systolic']
            delta_diastolic = current_metrics['avg_diastolic'] - average_metrics['avg_diastolic']
            st.metric("Blood Pressure", 
                     f"{current_metrics['avg_systolic']:.1f}/{current_metrics['avg_diastolic']:.1f}", 
                     f"{delta_systolic:+.1f}/{delta_diastolic:+.1f} vs avg")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Heart Rate Throughout the Day")
            fig_hr = px.line(daily_data, x='Timestamp (IST)', y='HR',
                           title='Heart Rate Variation')
            st.plotly_chart(fig_hr, use_container_width=True)
            
            st.subheader("Step Count Progress")
            fig_steps = px.bar(daily_data, x='Timestamp (IST)', y='Steps',
                             title='Steps per 5-minute Interval')
            st.plotly_chart(fig_steps, use_container_width=True)
        
        with col2:
            st.subheader("Blood Pressure Readings")
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(x=daily_data['Timestamp (IST)'], 
                                      y=daily_data['BP Systolic'],
                                      name='Systolic'))
            fig_bp.add_trace(go.Scatter(x=daily_data['Timestamp (IST)'], 
                                      y=daily_data['BP Diastolic'],
                                      name='Diastolic'))
            fig_bp.update_layout(title='Blood Pressure Throughout the Day')
            st.plotly_chart(fig_bp, use_container_width=True)
            
            st.subheader("SpO2 Levels")
            fig_spo2 = px.line(daily_data, x='Timestamp (IST)', y='SpO2 (%)',
                             title='SpO2 Variation')
            fig_spo2.add_hline(y=95, line_dash="dash", line_color="red",
                             annotation_text="Minimum Safe Level")
            st.plotly_chart(fig_spo2, use_container_width=True)
        
        # Activity and Sleep Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Today's Activity Distribution")
            activity_dist = daily_data['Activity Level'].value_counts()
            fig_activity = px.pie(
                values=activity_dist.values,
                names=activity_dist.index,
                title='Current Day Activity Distribution'
            )
            st.plotly_chart(fig_activity, use_container_width=True)
        
        with col2:
            if all(col in daily_data.columns for col in ['REM Sleep', 'Deep Sleep', 'Light Sleep']):
                st.subheader("Sleep Distribution")
                sleep_data = {
                    'Sleep Stage': ['REM Sleep', 'Deep Sleep', 'Light Sleep'],
                    'Hours': [
                        daily_data['REM Sleep'].mean(),
                        daily_data['Deep Sleep'].mean(),
                        daily_data['Light Sleep'].mean()
                    ]
                }
                sleep_df = pd.DataFrame(sleep_data)
                fig_sleep = px.bar(sleep_df, x='Sleep Stage', y='Hours',
                                 title='Sleep Stage Distribution',
                                 color='Sleep Stage')
                st.plotly_chart(fig_sleep, use_container_width=True)
            else:
                st.info("Sleep data is not available in the current dataset")

        # Show raw data option
        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(daily_data)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure your CSV file is in the correct format and contains all required columns.")

def show_recommendations():
    st.title("Personalized Health Recommendations")
    
    try:
        # Load data
        df = load_data()
        
        # Sidebar for date filtering
        st.sidebar.header("Filters")
        dates = df['Date'].unique()
        selected_date = st.sidebar.selectbox("Select Date", dates)
        
        # Filter data
        daily_data = df[df['Date'] == selected_date]
        
        # Calculate metrics
        current_metrics = calculate_health_metrics(daily_data)
        average_metrics = calculate_health_metrics(df)
        
        # Display current metrics summary
        st.header("Current Health Metrics Summary")
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Steps", f"{current_metrics['avg_steps']:,.0f}")
            st.metric("Heart Rate", f"{current_metrics['avg_hr']:.1f} bpm")
        
        with cols[1]:
            st.metric("Blood Pressure", f"{current_metrics['avg_systolic']:.1f}/{current_metrics['avg_diastolic']:.1f}")
            st.metric("SpO2", f"{current_metrics['avg_spo2']:.1f}%")
        
        # Only show sleep metrics if available
        if all(key in current_metrics for key in ['rem_sleep', 'deep_sleep', 'light_sleep']):
            with cols[2]:
                total_sleep = (current_metrics['rem_sleep'] + 
                             current_metrics['deep_sleep'] + 
                             current_metrics['light_sleep'])
                st.metric("Total Sleep", f"{total_sleep:.1f} hours")
                st.metric("Deep Sleep", f"{current_metrics['deep_sleep']:.1f} hours")
        
        # Generate and display recommendations
        st.header("AI-Powered Health Recommendations")
        with st.spinner("Generating personalized recommendations..."):
            recommendations = get_recommendations(current_metrics, average_metrics)
            st.markdown(recommendations)
        
        # Historical trends
        st.header("Health Metrics Trends")
        metrics = ["Steps", "Heart Rate", "Blood Pressure", "SpO2"]
        if all(col in df.columns for col in ['REM Sleep', 'Deep Sleep', 'Light Sleep']):
            metrics.append("Sleep Duration")
            
        metric_option = st.selectbox(
            "Select Metric to View Trend",
            metrics
        )
        
        # Create trend visualization based on selection
        if metric_option == "Steps":
            fig = px.line(df, x='Date', y='Total Day Steps', title='Steps Trend')
        elif metric_option == "Heart Rate":
            fig = px.line(df, x='Date', y='HR', title='Heart Rate Trend')
        elif metric_option == "Blood Pressure":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BP Systolic'], name='Systolic'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BP Diastolic'], name='Diastolic'))
            fig.update_layout(title='Blood Pressure Trend')
        elif metric_option == "SpO2":
            fig = px.line(df, x='Date', y='SpO2 (%)', title='SpO2 Trend')
        elif metric_option == "Sleep Duration" and 'REM Sleep' in df.columns:
            df['Total Sleep'] = df['REM Sleep'] + df['Deep Sleep'] + df['Light Sleep']
            fig = px.line(df, x='Date', y='Total Sleep', title='Total Sleep Duration Trend')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure your CSV file is in the correct format and contains all required columns.")


##################################################################################################################################################


class FitnessRecommendationSystem:
    def __init__(self):
        self.calorie_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, df):
        activity_map = {
            'Sedentary': 0,
            'Lightly Active': 1,
            'Moderately Active': 2,
            'Very Active': 3,
            'Extra Active': 4
        }
        df['activity_level_encoded'] = df['activity_level'].map(activity_map)
        df['gender_encoded'] = self.label_encoder.fit_transform(df['gender'])
        
        features = ['age', 'height', 'weight', 'bmi', 'activity_level_encoded', 'gender_encoded']
        X = df[features]
        y_calories = df['target_calories']
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_calories, features
    
    def train_models(self, X, y_calories):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_calories, test_size=0.2, random_state=42
        )
        
        self.calorie_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.calorie_model.fit(X_train, y_train)
        
        cal_pred = self.calorie_model.predict(X_test)
        
        return {'calorie_rmse': np.sqrt(mean_squared_error(y_test, cal_pred))}
    
    def get_recommendations(self, user_data, goal_type, rate_per_week):
        user_features = pd.DataFrame([user_data])
        activity_map = {
            'Sedentary': 0,
            'Lightly Active': 1,
            'Moderately Active': 2,
            'Very Active': 3,
            'Extra Active': 4
        }
        user_features['activity_level_encoded'] = user_features['activity_level'].map(activity_map)
        user_features['gender_encoded'] = self.label_encoder.transform(user_features['gender'])
        
        features = ['age', 'height', 'weight', 'bmi', 'activity_level_encoded', 'gender_encoded']
        user_scaled = self.scaler.transform(user_features[features])
        
        base_calories = self.calorie_model.predict(user_scaled)[0]
        weekly_calorie_adjustment = rate_per_week * 7700  # 1 kg = 7700 calories
        daily_calorie_adjustment = weekly_calorie_adjustment / 7
        
        target_calories = base_calories - daily_calorie_adjustment if goal_type == "Lose Weight" else base_calories + daily_calorie_adjustment
            
        return self.generate_recommendations(base_calories, target_calories, goal_type, rate_per_week, user_data)
    
    def generate_exercise_plan(self, activity_level, goal_type):
        plans = {
            'Sedentary': {
                'Lose Weight': [
                    "Week 1-2: 30 min walking 5x/week",
                    "Week 3-4: 45 min walking + basic bodyweight exercises",
                    "Week 5+: Add 2 strength training days",
                    "Daily step goal: Start at 7,000, increase by 1,000 weekly",
                    "Core work: 15 min, 3x/week",
                    "Weekend: Light yoga or stretching"
                ],
                'Gain Weight': [
                    "3x/week full-body strength training",
                    "Focus on compound movements",
                    "20-30 min light cardio for health",
                    "Progressive overload: increase weights 2-5% weekly",
                    "Rest days: Light stretching or walking"
                ]
            },
            'Moderately Active': {
                'Lose Weight': [
                    "Strength Training 3-4x/week:",
                    "- Upper body: Mon/Thu",
                    "- Lower body: Tue/Fri",
                    "HIIT: 2x/week, 20-30 min",
                    "Cardio: 45 min moderate intensity 2x/week",
                    "Core work: 3x/week",
                    "Active recovery: yoga or swimming"
                ],
                'Gain Weight': [
                    "4-day split routine:",
                    "- Chest/Triceps",
                    "- Back/Biceps",
                    "- Legs/Core",
                    "- Shoulders/Arms",
                    "Progressive overload focus",
                    "Limit cardio to 20 min 2x/week"
                ]
            }
        }
        return plans.get(activity_level, {}).get(goal_type, ["Custom plan needed"])



    



    
    def generate_recommendations(self, base_calories, target_calories, goal_type, rate_per_week, user_data):
        weight_kg = user_data['weight']
        
        # Define protein ranges based on activity level (min, max)
        activity_protein_ranges = {
            'Sedentary': (0.8, 1.0),
            'Lightly Active': (1.0, 1.4),
            'Moderately Active': (1.4, 1.7),
            'Very Active': (1.7, 2.0),
            'Extra Active': (2.0, 2.2)  # Same as Very Active
        }
        
        # Get base protein range from activity level
        base_protein_range = activity_protein_ranges[user_data['activity_level']]
        
        # Adjust protein range based on goal
        if goal_type == "Gain Weight" and user_data['activity_level'] in ['Very Active', 'Extra Active']:
            protein_range = (2.0, 2.2)  # Special range for active people trying to gain weight
        else:
            protein_range = base_protein_range
        
        # Calculate actual protein amounts
        min_protein = int(weight_kg * protein_range[0])
        max_protein = int(weight_kg * protein_range[1])
        
        # Calculate protein per meal (assuming 4 meals per day)
        min_protein_per_meal = int(min_protein / 4)
        max_protein_per_meal = int(max_protein / 4)
        
        recommendations = {
            'base_calories': int(base_calories),
            'target_calories': int(target_calories),
            'exercise_plan': self.generate_exercise_plan(user_data['activity_level'], goal_type),
            'nutrition_tips': [
                f"Daily calories: {int(target_calories)} kcal",
                f"Daily protein range: {min_protein}-{max_protein}g ({protein_range[0]:.1f}-{protein_range[1]:.1f}g per kg bodyweight)",
                f"Protein per meal: {min_protein_per_meal}-{max_protein_per_meal}g",
                f"Water: {int(weight_kg * 35)}ml/day",
                "Protein timing:",
                "  • Breakfast: 20% of daily target",
                "  • Lunch: 30% of daily target",
                "  • Post-workout: 30% of daily target",
                "  • Dinner: 20% of daily target",
                "Carbohydrate recommendations:",
                f"  • Training days: {int(target_calories * 0.45 / 4)}g-{int(target_calories * 0.55 / 4)}g",
                f"  • Rest days: {int(target_calories * 0.35 / 4)}g-{int(target_calories * 0.45 / 4)}g",
                "Fat recommendations:",
                f"  • Minimum: {int(target_calories * 0.20 / 9)}g per day",
                f"  • Maximum: {int(target_calories * 0.30 / 9)}g per day",
                "Supplement recommendations:",
                "  • Whey/casein protein if needed to meet targets",
                "  • BCAAs for fasted training",
                "  • Creatine monohydrate 5g daily"
            ],
            'weekly_goals': [
                f"Target weight change: {rate_per_week} kg/week",
                "Track measurements weekly:",
                "  • Morning body weight",
                "  • Waist circumference",
                "  • Body fat % (if available)",
                "  • Key body measurements",
                "Progress photos bi-weekly",
                "Strength progression tracking:",
                "  • Log main lifts",
                "  • Track volume progression",
                "Recovery monitoring:",
                "  • Sleep quality (7-9 hours)",
                "  • Muscle soreness levels",
                "  • Energy levels",
                "  • Hydration status"
            ]
        }
        return recommendations
    
    
    


##############################################################################################################################################


class RealTimeFitnessDashboard:
    def __init__(self):
        self.credentials = None
    
    def clear_credentials(self):
        if os.path.exists('token.pickle'):
            os.remove('token.pickle')
        st.session_state.credentials = None
        st.experimental_rerun()
    
    def render_dashboard(self):
        st.title("Google Fit Real-Time Health Dashboard")
        
        # Credential management
        if 'credentials' not in st.session_state:
            st.session_state.credentials = None
        
        if st.session_state.credentials is not None:
            if st.sidebar.button('Sign Out'):
                self.clear_credentials()
        
        try:
            # Authentication
            if st.session_state.credentials is None:
                with st.spinner('Authenticating with Google Fit...'):
                    st.session_state.credentials = self._get_credentials()
                st.success('Authentication successful!')
            
            fitness_data = self._get_fitness_data(st.session_state.credentials)
            
            # Sidebar settings
            st.sidebar.title("Settings")
            days_steps = st.sidebar.slider("Days of Steps Data", 1, 30, 7)
            days_vitals = st.sidebar.slider("Days of Vitals Data", 1, 7, 1)
            real_time = st.sidebar.checkbox("Enable Real-time Updates", value=False)
            update_interval = st.sidebar.number_input("Update Interval (seconds)", min_value=30, value=60) if real_time else None
            
            # Create placeholder for real-time charts
            vitals_container = st.empty()
            
            while True:
                # Steps Data
                st.header("Daily Steps")
                with st.spinner('Fetching steps data...'):
                    steps_df = fitness_data.get_steps_data(days=days_steps)
                
                fig_steps = px.bar(
                    steps_df,
                    x='date',
                    y='steps',
                    title='Daily Steps Count'
                )
                st.plotly_chart(fig_steps, use_container_width=True)
                
                # Vitals Data in a single container
                with vitals_container.container():
                    self._render_vitals_charts(fitness_data, days_vitals, steps_df)
                
                if not real_time:
                    break
                    
                time.sleep(update_interval)
        
        except RefreshError:
            self.clear_credentials()
            st.error("Authentication expired. Please sign in again.")
        except Exception as e:
            self._handle_error(e)
    
    def _render_vitals_charts(self, fitness_data, days_vitals, steps_df):
        # Heart Rate
        with st.spinner('Fetching heart rate data...'):
            hr_df = fitness_data.get_heart_rate_data(days=days_vitals)
        
        if not hr_df.empty:
            fig_hr = px.line(
                hr_df,
                x='timestamp',
                y='heart_rate',
                title='Heart Rate Over Time (BPM)'
            )
            st.plotly_chart(fig_hr, use_container_width=True)
        
        # # Blood Oxygen
        # with st.spinner('Fetching SpO2 data...'):
        #     spo2_df = fitness_data.get_blood_oxygen_data(days=days_vitals)
        
        # if not spo2_df.empty:
        #     fig_spo2 = px.line(
        #         spo2_df,
        #         x='timestamp',
        #         y='spo2',
        #         title='Blood Oxygen Saturation (SpO2) %'
        #     )
        #     st.plotly_chart(fig_spo2, use_container_width=True)
        
        # # Blood Pressure
        # with st.spinner('Fetching blood pressure data...'):
        #     bp_df = fitness_data.get_blood_pressure_data(days=days_vitals)
        
        # if not bp_df.empty:
        #     fig_bp = px.line(
        #         bp_df,
        #         x='timestamp',
        #         y=['systolic', 'diastolic'],
        #         title='Blood Pressure Over Time (mmHg)'
        #     )
        #     st.plotly_chart(fig_bp, use_container_width=True)
        
        # Summary Statistics
        # self._render_summary_metrics(steps_df, hr_df, spo2_df, bp_df)
        self._render_summary_metrics(steps_df, hr_df)
    
    # def _render_summary_metrics(self, steps_df, hr_df, spo2_df, bp_df):
    def _render_summary_metrics(self, steps_df, hr_df):
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Average Daily Steps",
                f"{int(steps_df['steps'].mean()):,}",
                f"{int(steps_df['steps'].iloc[-1] - steps_df['steps'].mean()):+,}"
            )
        
        with cols[1]:
            if not hr_df.empty:
                st.metric(
                    "Current Heart Rate",
                    f"{int(hr_df['heart_rate'].iloc[-1])} BPM",
                    f"{int(hr_df['heart_rate'].iloc[-1] - hr_df['heart_rate'].mean()):+} BPM"
                )
        
        # with cols[2]:
        #     if not spo2_df.empty:
        #         st.metric(
        #             "Current SpO2",
        #             f"{spo2_df['spo2'].iloc[-1]:.1f}%",
        #             f"{(spo2_df['spo2'].iloc[-1] - spo2_df['spo2'].mean()):.1f}%"
        #         )
        
        # with cols[3]:
        #     if not bp_df.empty:
        #         st.metric(
        #             "Current Blood Pressure",
        #             f"{int(bp_df['systolic'].iloc[-1])}/{int(bp_df['diastolic'].iloc[-1])}",
        #             f"{int(bp_df['systolic'].iloc[-1] - bp_df['systolic'].mean()):+}/{int(bp_df['diastolic'].iloc[-1] - bp_df['diastolic'].mean()):+}"
        #         )
    
    def _handle_error(self, e):
        st.error(f"An error occurred: {str(e)}")
        st.info("""
        Please make sure you have:
        1. Created a Google Cloud Project
        2. Enabled the Fitness API
        3. Created OAuth 2.0 credentials
        4. Added the correct redirect URIs in Google Cloud Console
           - Add http://localhost:* to allow any port
        5. Downloaded and saved the credentials as 'credentials.json'
        """)
        
        if st.button('Clear Authentication'):
            self.clear_credentials()
    
    def _get_credentials(self):
        # Placeholder for get_credentials() method from config.py
        # You'll need to import or implement this method
        from config import get_credentials
        return get_credentials()
    
    def _get_fitness_data(self, credentials):
        # Placeholder for FitnessData class from fitness_data.py
        # You'll need to import or implement this class
        from fitness_data import FitnessData
        return FitnessData(credentials)


###########################################################################################################################################





def main():
    # Create tabs
    tabs = st.tabs(["Dashboard", "Goal Setting", "Real-time Tracking"])
    
    with tabs[0]:   
        # Create a navigation sidebar
        page = st.sidebar.radio("Go to", ["Dashboard", "Recommendations"])
        
        if page == "Dashboard":
            show_dashboard()
        else:
            show_recommendations()
    
    with tabs[1]:   
        st.title("Advanced Fitness Recommendation System")
        
        try:
            df = pd.read_csv('bmi_dataset.csv')
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return
        
        recommender = FitnessRecommendationSystem()
        X_scaled, y_calories, features = recommender.prepare_data(df)
        recommender.train_models(X_scaled, y_calories)
        
        st.subheader("Your Information")
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.1)
            height = st.number_input("Height (cm)", 140.0, 220.0, 170.0, step=0.1)
            age = st.number_input("Age", 18, 100, 30)
            
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            activity_level = st.selectbox("Activity Level", 
                                        ["Sedentary", "Lightly Active", "Moderately Active", 
                                        "Very Active", "Extra Active"])
        
        st.subheader("Goal Setting")
        goal_type = st.selectbox("Select your goal:", ["Lose Weight", "Gain Weight"])
        
        rate_options = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)] if goal_type == "Lose Weight" else [round(x, 1) for x in np.arange(0.1, 0.6, 0.1)]
        
        rate_per_week = st.select_slider(
            "Target rate (kg per week):",
            options=rate_options,
            value=0.5
        )
        
        bmi = weight / ((height/100) ** 2)
        
        if st.button("Generate Personalized Plan"):
            user_data = {
                'age': age,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'activity_level': activity_level,
                'gender': gender
            }
            
            recommendations = recommender.get_recommendations(user_data, goal_type, rate_per_week)
            
            st.subheader("Your Personalized Plan")
            st.write(f"Maintenance Calories: {recommendations['base_calories']} kcal/day")
            st.write(f"Target Calories: {recommendations['target_calories']} kcal/day")
            
            st.subheader("Exercise Plan")
            for exercise in recommendations['exercise_plan']:
                st.write(f"• {exercise}")
                
            st.subheader("Nutrition Guidelines")
            for tip in recommendations['nutrition_tips']:
                st.write(f"• {tip}")
                
            st.subheader("Weekly Targets")
            for goal in recommendations['weekly_goals']:
                st.write(f"• {goal}")
    
    with tabs[2]:
        dashboard = RealTimeFitnessDashboard()
        dashboard.render_dashboard()

if __name__ == "__main__":
    main()