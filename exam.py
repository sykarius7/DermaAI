import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
from pathlib import Path
import streamlit_authenticator as stauth 
import google.generativeai as genai
from dotenv import load_dotenv
st.set_page_config(page_title="Skin Disease Classification", page_icon=":microscope:", layout="wide")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- USER AUTHENTICATION ---
names=["Ahobilesha","Aditya"]
usernames=["aho","adi"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "dermAI", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

class_names = [
    'Acne and Rosacea Photos', 'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos', 'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections'
]

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image_class(model, image):
    image = np.array(image.resize((400, 400))) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    predicted_class = np.argmax(pred, axis=1)
    return class_names[predicted_class[0]]

model = genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

# Function to get response from Gemini model
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def chatbot():
    st.title("Ask  DermAI anything about skin conditions!")
    input_text = st.text_input("Input: ", key="input")
    submit_button = st.button("Ask the question")

    user_chat_history_key = f"{username}_chat_history"
    chat_history = st.session_state.get(user_chat_history_key, [])
    if submit_button and input_text:
        response = get_gemini_response(input_text)
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)

        # Update the chat history with the current input and response
        chat_history.append(("You", input_text))
        for chunk in response:
            chat_history.append(("Bot", chunk.text))
        st.session_state[user_chat_history_key] = chat_history

    if chat_history:
        st.markdown('<h3 style="color: yellow;">The Chat History is</h3>', unsafe_allow_html=True)
        for role, text in chat_history:
            if role == "You":
                st.markdown(f'<span style="color: blue;">You: {text}</span>', unsafe_allow_html=True)
            elif role == "Bot":
                st.markdown(f'<span style="color: green;">Bot: {text}</span>', unsafe_allow_html=True)

def main():
    if authentication_status:
        authenticator.logout("Logout", "sidebar")
        option = st.sidebar.radio("Navigation", [ "DermAI","ChatBot"], index=1)
        if option == "ChatBot":
            chatbot()
        elif option == "DermAI":
            st.title('Skin Disease Classification')

            st.write("This app is a tool for predicting skin disease types. Upload an image to see the predicted class. Please note that this is a tool for educational purposes and the prediction is not a diagnosis.")

            uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
            st.subheader("OR")
            camera_file=st.camera_input("Take a photo")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                st.write('')

                st.write('Classifying...')

                model = load_model('bestmodel1.h5')
                predicted_class = predict_image_class(model, image)

                st.success(f'Predicted Class: {predicted_class}')
            if camera_file is not None:
                image1 = Image.open(camera_file)
                st.success("Photo uploaded successfully")
                st.image(image1, caption='Uploaded Image', use_column_width=True)
                st.write('')

                st.write('Classifying...')

                model = load_model('bestmodel1.h5')
                predicted_class = predict_image_class(model, image1)

                st.success(f'Predicted Class: {predicted_class}')

if __name__ == '__main__':
    main()
