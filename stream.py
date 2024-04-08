from PIL import Image
import requests
import streamlit as st
import sounddevice as sd
import soundfile as sf
import tempfile
import io
import time
import pyttsx3

API_TEXT_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
API_IMAGE_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_IMAGE_TO_TEXT_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
API_LANGUAGE_ID_URL = "https://api-inference.huggingface.co/models/speechbrain/lang-id-voxlingua107-ecapa"
API_VOICE_ASSISTANT_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_text(payload):
    response = requests.post(API_TEXT_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def query_image(payload):
    response = requests.post(API_IMAGE_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.content

def query_image_to_text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_IMAGE_TO_TEXT_URL, headers=headers, data=data)
    response.raise_for_status()
    return response.json()

def query_language_identification(audio_data):
    response = requests.post(API_LANGUAGE_ID_URL, headers=headers, data=audio_data)
    while response.status_code == 503:
        print("Model still loading. Waiting for 10 seconds...")
        time.sleep(10)
        response = requests.post(API_LANGUAGE_ID_URL, headers=headers, data=audio_data)
    response.raise_for_status()
    return response.json()

def query_voice_assistant(payload):
    response = requests.post(API_VOICE_ASSISTANT_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def record_audio(seconds=5, sample_rate=44100, format='flac'):
    st.write("Recording...")
    audio_data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    st.write("Finished recording.")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        filename = tmp_file.name
        sf.write(filename, audio_data, sample_rate, format=format)
    
    return filename, sample_rate

def score_to_percentile(score):
    percentile = score * 100
    return percentile

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("Chatbot with Text, Image, and Voice Assistant")

    st.sidebar.header("Choose Input Method")
    input_method = st.sidebar.radio("Select Input Method", ("Text", "Image", "Image to Text", "Language Identification", "Voice Assistant"))

    if input_method == "Text":
        user_input = st.text_input("Enter your message here:")
        
        if "hi" in user_input.lower():
            st.text("Bot: Hi there!")
        elif "bye" in user_input.lower():
            st.text("Bot: Goodbye!")
        
        if st.button("Send"):
            st.text("You: " + user_input)
            payload = {"inputs": user_input}
            try:
                output = query_text(payload)
                bot_response = output[0]['generated_text'].strip()
                st.text("Bot: " + bot_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


    elif input_method == "Image":
        image_prompt = st.text_input("Enter image prompt:")
        if st.button("Generate Image"):
            payload = {"inputs": image_prompt}
            try:
                image_bytes = query_image(payload)
                img = Image.open(io.BytesIO(image_bytes))
                st.image(img, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif input_method == "Image to Text":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if st.button("Generate Text"):
            try:
                if uploaded_file is not None:
                    with open("temp_image.jpg", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    output = query_image_to_text("temp_image.jpg")
                    st.write(output)
                    generated_text = output.get("result", "No result found")
                    st.text("Generated Text: " + generated_text)
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    elif input_method == "Language Identification":
        record_button = st.button("Record Audio")
        if record_button:
            try:
                audio_path, sample_rate = record_audio()
                st.audio(audio_path, format="audio/wav")
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                output = query_language_identification(audio_data)
                for item in output:
                    percentile_score = score_to_percentile(item['score'])
                    if percentile_score >= 10:
                        st.write(f"{item['label'].split(':')[1]} - {percentile_score}%")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif input_method == "Voice Assistant":
        voice_input = st.text_input("Speak to the Voice Assistant:")
        
        if "hi" in voice_input.lower():
            st.text("Voice Assistant: Hi there!")
            speak_text("Hi there!")
        elif "bye" in voice_input.lower():
            st.text("Voice Assistant: Goodbye!")
            speak_text("Goodbye!")
        
        if st.button("Speak"):
            payload = {"inputs": voice_input}
            try:
                output = query_voice_assistant(payload)
                bot_response = output[0]['generated_text'].strip()
                st.text("Voice Assistant: " + bot_response)
                speak_text(bot_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
