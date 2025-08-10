import os
import speech_recognition as sr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import requests
from math import radians, cos, sin, asin, sqrt

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

class TriageResponse(BaseModel):
    urgency: str = Field(description="Classify urgency: 'Emergency', 'Urgent', or 'Routine'.")
    possible_causes: List[str] = Field(description="A list of 2-3 potential, general causes.")
    suggested_specialty: str = Field(description="The title of the medical specialist to consult (e.g., 'Cardiologist', 'Neurologist', 'General Physician').")
    self_care_tips: Optional[List[str]] = Field(description="Safe, actionable self-care tips for non-emergencies.")
    recommended_tests: Optional[List[str]] = Field(description="Common diagnostic tests a doctor might recommend.")
    explanation: str = Field(description="A brief, clear explanation for the overall assessment.")

def load_ai_model():
    """Loads and returns the initialized Gemini model and the response parser."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY for Gemini not found!")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=TriageResponse)
    return llm, parser

def get_ai_triage_analysis(llm, parser, patient_text: str, language: str):
    """Gets the structured triage analysis from the LLM."""
    prompt_template = """
    You are 'VoiceDoc', a precise medical triage AI. Analyze the patient's statement in {language} and output a valid JSON object in English.
    **CRITICAL INSTRUCTION: Your output MUST be a single, valid JSON object. Do not add any text before or after it.**
    {format_instructions}
    **Patient's statement ({language}):** "{patient_text}"
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["patient_text", "language"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({"patient_text": patient_text, "language": language})
    except Exception as e:
        return f"Error processing AI response: {e}"

def transcribe_audio(audio_file_path: str, language_code: str) -> str:
    """Transcribes an audio file using the free Google Web Speech API."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        return "Error: Could not understand the audio."
    except sr.RequestError:
        return "Error: API request failed. Check internet."

def find_nearby_places_google(latitude, longitude, query, radius=10000):
    """Finds nearby places using the Google Maps Places API."""
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY not found!")
        
    api_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f"{latitude},{longitude}",
        'radius': radius,
        'keyword': query,
        'key': GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        places = []
        if data.get('results'):
            for place in data['results']:
                loc = place.get('geometry', {}).get('location', {})
                # Haversine distance calculation
                R = 6371; lat1, lon1, lat2, lon2 = map(radians, [latitude, longitude, loc.get('lat', 0), loc.get('lng', 0)])
                dLat, dLon = lat2 - lat1, lon2 - lon1
                a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
                dist = 2 * R * asin(sqrt(a))
                
                places.append({
                    "Name": place.get('name', 'N/A'),
                    "Address": place.get('vicinity', 'N/A'),
                    "Rating": place.get('rating', 'N/A'),
                    "Distance (km)": dist
                })
        
        if not places: return pd.DataFrame()
        return pd.DataFrame(places).sort_values('Distance (km)')
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Google Maps API: {e}"
    except Exception as e:
        return f"An error occurred while parsing Google Maps data: {e}"