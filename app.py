import warnings
import librosa
import streamlit as st
from joblib import dump, load
from audio_analysis import audio_signals
from audio_processing import extract_features
# from audio_record import record_audio

emotion = st.selectbox(
    'Select a Emotion',
    ('Angry', 'Disgust', 'Fear',
       'Happy', 'Neutral', 'Pleasant_Suprise','Sad'))

st.write('You selected:', emotion)


if emotion == 'Angry':
    file_path = st.selectbox(
    'Select a Audio file',
    ('raw_audio_data/Angry/OAF_boat_angry.wav', 
    'raw_audio_data/Angry/OAF_boat_angry.wav', 
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav',
    'raw_audio_data/Angry/OAF_boat_angry.wav'))
elif emotion == 'Disgust':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\OneDrive\Desktp\app\raw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\aw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\raw_audio_data\Angy\OAF_calm_angry.wav'))
elif emotion == 'Fear':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\OneDrive\Desktop\app\raw_audgry\OAF_calm_angry.wav', 
    r'C:\Users\shok\OneDrive\Desktop\app\raw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Usersashok\OneDrive\Desktop\app\raw_audio_data\Angry\OAF_calm_angry.wav'))
elif emotion == 'Happy':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\Oneraw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\asrive\Desktop\app\raw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\raw_aAF_calm_angry.wav'))
elif emotion == 'Neutral':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\OneDrive\Deaudio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\raw_audio_data\Angryngry.wav', 
    r'C:\Users\ashok\Ontop\app\raw_audio_data\Angry\OAF_calm_angry.wav'))
elif emotion == 'Pleasant_Suprise':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\OneDrive\Desktop\app\raw_audio_data\Angrylm_angry.wav', 
    r'C:\Users\ashok\OneDesktop\app\raw_audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\raw_audio_data\A_calm_angry.wav'))
elif emotion == 'Sad':
    file_path = st.selectbox(
    'Select a Audio file',
    (r'C:\Users\ashok\OneDrive\Desktop\app\raw_audiAngry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\audio_data\Angry\OAF_calm_angry.wav', 
    r'C:\Users\ashok\OneDrive\Desktop\app\udio_data\Angry\OAF_calm_angry.wav'))
else:
    st.subheader('File not found')


audio_signals(file_path)

audio_data, sampling_rate = librosa.load(file_path)
st.audio(audio_data, sample_rate=sampling_rate)

model_path = 'audio_classifier_model.joblib'
model = load(model_path)

audio = file_path
print(audio)
extracted_features = extract_features(audio).reshape(1, -1)
# extracted_features = x_test[112].reshape(1, -1)
y_predict = model.predict(extracted_features)
labels_list = ['Fear', 'Angry', 'Neutral', 'Sad', 'Pleasant_Suprised', 'Disgust', 'Happy']
encoded_label = [2, 0, 4, 6, 5, 1, 3]

labels = {}
for label, prediction in zip(encoded_label, labels_list):
    labels[label] = prediction
if y_predict[0] in labels.keys():
    st.write('Predicted Class:', labels[y_predict[0]])
