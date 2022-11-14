
import streamlit as st
import requests
import json
from PIL import Image
from base64 import b64decode

image = Image.open('Logo.png')
st.image(image, width = 100)
st.title("ViShield")
st.write("Vision Shield, An app that helps you to filter out snakes from your images and videos")
st.caption("For more information, visit my [github](https://github.com/erjieyong?tab=repositories) or contact me directly at [erjieyong@gmail.com](mailto:erjieyong@gmail.com)")

with st.form(key='my_form'):
    url = st.text_input("Image / Video URL", placeholder="Please enter image url")
    st.caption("Please ensure that your url ends with image or video formats such as .jpg, .png, .gif, .mp4")
    submit = st.form_submit_button(label='Submit')

if submit:
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tiff', 'dng', 'webp', 'mpo'] #acceptable image formats
    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'm4v', 'wmv', 'mkv', 'gif', 'webm'] #acceptable video formats
    file_typ = url.split(".")[-1] #get the file type

    if len(url) == 0 or (file_typ not in img_formats and file_typ not in vid_formats):
        st.write("Please enter a valid url ending with the correct image or video formats")
    else:
        with st.spinner('🪄 ✨Gathering magic dusts...✨'):
            api_url = 'https://jy-dsi-capstone-no755hevjq-as.a.run.app'
            # api_url = 'http://localhost:8080'
            api_route = '/predict'
            # api_stream = '/stream'


        # response = requests.post(f'{api_url}{api_route}', json=json.dumps(url), stream=True)
        # for chunk in response.iter_content(1024):
        #     print(chunk)

            
            response = requests.post(f'{api_url}{api_route}', json=json.dumps(url)) # json.dumps() converts dict to JSON
            response = response.json()

            # stream = requests.get(f'{api_url}{api_stream}', stream=True) # json.dumps() converts dict to JSON
            # print('attempting to connect to stream...')
            # for line in stream.iter_lines():
            #     if line:
            #         print(line)
            logs = None
            print(response['logs'])
            
            print(type(response['logs']))
            print(len(response['logs']))

            if response['type'] == "img":
                img = response['img'].encode('utf8')
                img = b64decode(img)
                st.image(img)
                st.subheader("Prediction Logs")
                st.text(response['logs'])
                st.write(response['prompt'])
            elif response['type'] == "vid":
                vid = response['vid'].encode('utf8')
                vid = b64decode(vid)
                st.video(vid)
                st.subheader("Prediction Logs")
                st.text(response['logs'])
                st.text("Video is encoded in VP90 .WEBM format as opposed to H264 due to licensing issue")
                st.text(response['prompt'])
                
