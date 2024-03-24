import os
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
       'Happy', 'Neutral', 'Pleasant_Suprised','Sad'))

st.write('You selected:', emotion)

dirs = os.listdir(f'raw_audio_data/{emotion}')

if emotion == 'Angry':

    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))

elif emotion == 'Disgust':
    
    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))
    
elif emotion == 'Fear':
    
    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))
    
elif emotion == 'Happy':
    
    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))
    
elif emotion == 'Neutral':
    
    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))
    
elif emotion == 'Pleasant_Suprise':
    
    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))
    
elif emotion == 'Sad':

    file_list = []
    for dir_list in dirs:
    	files = ('raw_audio_data/'+emotion+'/'+dir_list)
    	file_list.append(files)
        
    file_path = st.selectbox(
    'Select a Audio file',
    tuple(file_list))

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
    st.subheader('Predicted Class:', labels[y_predict[0]])
