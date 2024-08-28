import streamlit as st
import os
import speech_recognition as sr
import tempfile
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from gtts import gTTS
import base64
from pydub import AudioSegment

# Set page title and icon

st.title("LEARN ANYTHING!!!!!")

# Add a background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://w0.peakpx.com/wallpaper/291/658/HD-wallpaper-white-brick-wall-texture-white-brick-background-stone-texture-white-bricks.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: black; /* Set text color to black */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def text_to_speech(text):
    if not text:
        st.write("Can you say that again???")
        return
    
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    audio = AudioSegment.from_file("response.mp3", format="mp3")
    audio = audio.speedup(playback_speed=1.2)
    audio.export("response.mp3", format="mp3")
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def calculate_height(text):
    # Approximate the height based on text length
    num_lines = text.count("\n") + 1
    height_per_line = 40  # Set an appropriate line height
    return max(150, num_lines * height_per_line)

def main():
    # Set background image
    set_background("https://www.pexels.com/photo/assorted-books-on-shelf-1290141/")

    groq_api_key = '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'

    conversational_memory_length = 10
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    
    if 'is_teaching' not in st.session_state:
        st.session_state.is_teaching = False

    if 'recorded_text' not in st.session_state:
        st.session_state.recorded_text = ""

    # Display the recognized text at the top with default text
    default_text = "HAPPY LEARNING...."
    text_to_display = st.session_state.recorded_text if st.session_state.recorded_text else default_text
    st.text_area("Now teaching about", text_to_display, height=calculate_height(text_to_display))

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-70b-8192')

    recognizer = sr.Recognizer()
    st.write("Click the button to record your topic or respond to the LLM.")

    # Place the button below the first text box
    if st.button("tell me"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        
        with sr.Microphone() as source:
            st.write("Recording...")
            audio = recognizer.listen(source)
            st.write("Recording stopped.")

            with open(temp_audio_path, "wb") as f:
                f.write(audio.get_wav_data())

        try:
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
                recorded_text = recognizer.recognize_google(audio)
                st.session_state.recorded_text = recorded_text
        except sr.UnknownValueError:
            st.write("Could not understand audio. Please record again.")
            recorded_text = None
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            recorded_text = None

        if recorded_text:
            if "stop" in recorded_text.lower():
                st.write("Teaching session ended.")
                st.session_state.is_teaching = False
                return

            if not st.session_state.is_teaching:
                st.session_state.is_teaching = True
                st.session_state.topic = recorded_text

                prep = f"I want to learn about {recorded_text}. Teach me step by step and ask if I have any doubts after each explanation."
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=prep),
                    MessagesPlaceholder(variable_name="chat_history"),
                ])

                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    memory=st.session_state.memory,
                )
                response = conversation.predict(human_input=prep)
                st.session_state.questions = [response]
                st.session_state.answers = []
                
                # Dynamically adjust height based on response length
                st.text_area("Model's Response", response, height=calculate_height(response), key="model_response")
                
                if response:
                    text_to_speech(response)
                else:
                    st.write("No response from the model.")
            else:
                st.session_state.answers.append(recorded_text)
                human_input = f"Based on the topic '{st.session_state.topic}', {recorded_text}, continue teaching and ask if I have any doubts."
                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=ChatPromptTemplate.from_messages([
                        SystemMessage(content=f"Continue teaching {st.session_state.topic}."),
                        MessagesPlaceholder(variable_name="chat_history"),
                    ]),
                    memory=st.session_state.memory,
                )
                response = conversation.predict(human_input=human_input)
                st.session_state.questions.append(response)
                
                # Dynamically adjust height based on response length
                st.text_area("Model's Response", response, height=calculate_height(response), key="model_response")
                
                if response:
                    text_to_speech(response)
                else:
                    st.write("No response from the model.")

        os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
