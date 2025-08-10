import streamlit as st
from dotenv import load_dotenv
import tempfile
import pandas as pd
import os
# Import our backend logic from main.py
from main import (
    load_ai_model,
    get_ai_triage_analysis,
    transcribe_audio,
    find_nearby_places_google,
    TriageResponse  # Import the class for type checking
)

# Import location components
from streamlit_geolocation import streamlit_geolocation
from geopy.geocoders import Nominatim

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="VoiceDoc AI Health Assistant", page_icon="🏥", layout="wide", initial_sidebar_state="auto"
)
# Load environment variables from .env file
load_dotenv()

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "user_location" not in st.session_state:
    st.session_state.user_location = None

# --- Load Models (this is cached by Streamlit) ---
try:
    llm, parser = load_ai_model()
except ValueError as e:
    st.error(str(e), icon="🚨")
    st.stop()


# --- 2. STREAMLIT USER INTERFACE ---
st.title("🏥 VoiceDoc – LIVE AI Health Assistant")
st.caption("Powered by Gemini 2.5 Flash & Google Maps Platform")

# Sidebar
st.sidebar.title("Configuration")
with st.sidebar:
    st.subheader("Step 1: Set Your Location")
    location_data = streamlit_geolocation()
    if location_data and location_data.get('latitude') is not None:
        st.session_state.user_location = (location_data['latitude'], location_data['longitude'])
        st.success(f"Location captured: ({location_data['latitude']:.4f}, {location_data['longitude']:.4f})")
    st.markdown("Or enter your city manually:")
    manual_city = st.text_input("Enter city name (e.g., Mumbai)", label_visibility="collapsed")
    if st.button("Find Location from City"):
        with st.spinner("Finding coordinates..."):
            try:
                geolocator = Nominatim(user_agent="voicedoc_app")
                location = geolocator.geocode(manual_city)
                if location:
                    st.session_state.user_location = (location.latitude, location.longitude)
                    st.success(f"Location set to {manual_city}: ({location.latitude:.4f}, {location.longitude:.4f})")
                else: st.error("Could not find this city.")
            except Exception as e: st.error(f"Geocoding error: {e}")
    st.divider()
    st.subheader("Step 2: Select Language")
    language_map = {"English (India)": "en-IN", "Hindi (हिन्दी)": "hi-IN"}
    selected_language_name = st.selectbox("Select Patient's Language", options=list(language_map.keys()), label_visibility="collapsed")
    language_code = language_map[selected_language_name]

# Main App Flow
st.subheader("Step 3: Upload Audio & Analyze")
if not st.session_state.user_location:
    st.warning("Please set your location in the sidebar to enable analysis.", icon="📍")
uploaded_file = st.file_uploader("Upload an audio recording:", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("Analyze Symptoms", type="primary", use_container_width=True, disabled=(st.session_state.user_location is None)):
        st.session_state.analysis_result = None
        with st.status(f"Analyzing in {selected_language_name}...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue()); temp_audio_path = tmp_file.name
            
            status.update(label="🎙️ Transcribing audio...")
            transcribed_text = transcribe_audio(temp_audio_path, language_code)
            os.remove(temp_audio_path)
            
            if "Error:" in transcribed_text:
                status.update(label="Transcription Failed.", state="error", expanded=True); st.error(transcribed_text, icon="🚫")
            else:
                status.update(label="🧠 Analyzing symptoms with Gemini Flash...")
                analysis_result = get_ai_triage_analysis(llm, parser, transcribed_text, selected_language_name)
                
                if isinstance(analysis_result, TriageResponse):
                    st.session_state.analysis_result = analysis_result
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                else:
                    st.session_state.analysis_result = None
                    status.update(label="AI Analysis Failed.", state="error", expanded=True); st.error(str(analysis_result), icon="🤖")

# --- Display Results ---
if isinstance(st.session_state.analysis_result, TriageResponse):
    result = st.session_state.analysis_result
    user_lat, user_lon = st.session_state.user_location
    st.subheader("AI Triage & Guidance Report", divider="blue")

    if result.urgency == "Emergency":
        st.error(f"**URGENCY: EMERGENCY**\n\nSeek immediate medical attention.", icon="🚨")
        with st.spinner("Searching for nearest hospitals with Google Maps..."):
            emergency_df = find_nearby_places_google(user_lat, user_lon, "hospital emergency")
            st.subheader("Nearest Emergency Services (from Google Maps)")
            if isinstance(emergency_df, pd.DataFrame) and not emergency_df.empty:
                st.table(emergency_df.style.format({'Rating':'{:.1f}', 'Distance (km)': '{:.1f}'}))
            else:
                st.warning("Could not find nearby emergency services.")
    
    tab1, tab2, tab3 = st.tabs(["**📋 Triage Summary**", "**❤️ Self-Care & Tests**", "**🧑‍⚕️ Find a Doctor**"])
    with tab1:
        st.metric(label="**Urgency Level**", value=result.urgency)
        st.markdown("**Possible Causes:**")
        for cause in result.possible_causes:
            st.markdown(f"- {cause}")
        with st.expander("**See AI's Full Reasoning**"):
            st.markdown(result.explanation)
    with tab2:
        st.markdown("**Recommended Self-Care Advice:**")
        if result.self_care_tips:
            for tip in result.self_care_tips:
                st.markdown(f"- {tip}")
        else:
            st.warning("No self-care is advised. Please consult a doctor.")
        st.markdown("**Suggested Diagnostic Steps:**")
        if result.recommended_tests:
            for test in result.recommended_tests:
                st.markdown(f"- {test}")
        else:
            st.info("The AI did not suggest specific tests.")
            
    with tab3:
        st.markdown(f"The AI recommends consulting a **{result.suggested_specialty}**.")
        with st.spinner(f"Finding nearest {result.suggested_specialty}s with Google Maps..."):
            doctors_df = find_nearby_places_google(user_lat, user_lon, result.suggested_specialty)
            st.subheader(f"Nearest {result.suggested_specialty}s (from Google Maps)")
            if isinstance(doctors_df, pd.DataFrame) and not doctors_df.empty:
                st.table(doctors_df.style.format({'Rating':'{:.1f}', 'Distance (km)': '{:.1f}'}))
            else:
                st.warning(f"Could not find any nearby doctors for the specialty '{result.suggested_specialty}'.")

st.divider()
st.warning("**Disclaimer:** This is a prototype and not a substitute for professional medical advice.", icon="⚠️")