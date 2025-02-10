import streamlit as st
from google_auth_oauthlib.flow import Flow
import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Google OAuth Configuration
CLIENT_SECRETS_FILE = "credentials.json"  # Ensure this is securely stored
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
]

REDIRECT_URI = "https://techvital.streamlit.app/"

# Store credentials in session state instead of a local file (Streamlit Cloud safe)
if "credentials" not in st.session_state:
    st.session_state.credentials = None

def get_credentials():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url

def fetch_credentials(auth_code):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=auth_code)
    return flow.credentials

# Streamlit UI
st.title("Google Login with OAuth in Streamlit")

# Step 1: Generate and display login button
if st.session_state.credentials is None:
    st.write("Login to connect your Google account:")
    auth_url = get_google_auth_link()
    st.markdown(f"[Click here to log in with Google]({auth_url})")

# Step 2: Handle OAuth response
query_params = st.experimental_get_query_params()
if "code" in query_params:
    auth_code = query_params["code"][0]
    credentials = fetch_credentials(auth_code)
    st.session_state.credentials = credentials

# Step 3: Display User Info after login
if st.session_state.credentials:
    credentials = st.session_state.credentials

    # Initialize Google API client
    service = build("oauth2", "v2", credentials=credentials)
    user_info = service.userinfo().get().execute()

    st.success(f"Logged in as: {user_info['name']} ({user_info['email']})")
    
    # Display user profile
    st.image(user_info["picture"], width=100)
