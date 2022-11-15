from flask import Flask, request, Response, stream_with_context
import os 
import sys
import time
import requests
import argparse
from base64 import b64encode

# to allow us to import and call predict.py from within yolov7
sys.path.insert(0, './yolov7/seg/segment/')
import predict

#to allow us to push images to roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="ENZ4UFQ5TkrSOtWuLtkb")
project = rf.workspace().project("snakes-oozrc")

api = Flask('ModelEndpoint')
#########################
##      FLASK API      ##
#########################
@api.route('/') 
def home(): 
    return {"message": "Hello!", "success": True}, 200

@api.route('/predict', methods = ['POST']) 
def make_predictions():
    url = request.get_json(force=True)
    url = url[1:-1] #get rid of " at the start and end of url
    file_typ = url.split(".")[-1] #get the file type
    file_link = 'target_image.'+file_typ 
    
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tiff', 'dng', 'webp', 'mpo'] #acceptable image formats
    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'm4v', 'wmv', 'mkv', 'gif'] #acceptable video formats

    # with the url, download and create the file locally
    r = requests.get(url)
    with open(file_link, 'wb') as f:
        f.write(r.content)
        
    # originally predict.py is supposed to be called in command line. But as we are calling it within another .py script, 
    # we need to create other own parser to pass it to predict.py
    # PREDICTION START---------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='./', help='save results to project/name')
    
    #check if file type is img or vid. If it is vid type, we will not save confidence level or text because we are not uploading them back to roboflow
    if file_typ in img_formats:
        opt = parser.parse_args(f"--save-txt --save-conf --weights ./Models/151122_0115/weights/best.pt --source target_image.{file_typ}".split())
    elif file_typ in vid_formats:
        opt = parser.parse_args(f"--weights ./Models/151122_0115/weights/best.pt --source target_image.{file_typ}".split())


    import logging
    logger = logging.getLogger('yolov5')
    handler = logging.FileHandler('predict.log')
    logger.addHandler(handler)

    predict.main(opt)

    # close the file handler
    logger.removeHandler(handler)
    handler.close()
 
    # PREDICTION END----------------------------------------------------------------------------------------------
    

    #Active learning
    #Read Results Files and Conditionally Upload

    #If my model has a confidence of less than 70% for a prediction, let's help it
    #out by uploading this image back to our dataset. Then we can add a ground truth
    #label to it so that it will be included in our next training run and future 
    #prediction results will improve.
    #Only do this for image files
    image_upload_txt = None
    if file_typ in img_formats:
        MIN_CONF_THRESHOLD = 0.7 
        predicted_image_dir = "./exp/labels"
        image_dir = "./"

        for i,txt_file in enumerate(os.listdir(predicted_image_dir)):
            print(i, txt_file)
            with open(os.path.join(predicted_image_dir,txt_file), 'r') as fid:
                for line in fid:
                    label, x1, y1, x2, y2, conf = line.split(" ")
                    conf = float(conf)
                    if conf < MIN_CONF_THRESHOLD:
                        image_name = txt_file[:-4]
                        image_upload_txt = f"Image has a low confidence prediction below 0.7, uploading to project SNAKES in roboflow for active learning"
                        print(image_upload_txt)
                        #Upload via Roboflow pip package
                        project.upload(os.path.join(image_dir,f'{file_link}'))
                    break
            
    # check for file type and send back the file to streamlit with the appropriate mimetype
    # take note that html5 player which streamlit uses only supports H264 encoding and not standard mp4v.
    # Hence we need to use mkv format which allows for H264 encoding. Changes were made directly to predict.py to allow for this encoding method
    # We also do encoding to b64 and utf8 instead of just `return send_file(filename, mimetype='video/mkv')` which is faster.
    # this is because we wanted to send more information together with the image/video
    # normally, we can send more information on top of send_file by using headers or cookies. But streamlit does not allow.
    # Hence we have to use this encoding and decoding method which takes slightly longer.
    if file_typ in img_formats:
        filename = "./exp/"+file_link
        # text that you want to send back together with file
        output = {
            "type" : "img",
            "prompt": image_upload_txt
        }
        #send logs
        with open("./predict.log", "r", encoding='utf-8') as log:
            content = log.read()
            output['logs'] = content
        #delete the predict.py log file
        if os.path.exists("predict.log"):
            os.remove("predict.log")
        #send image
        with open(filename, 'rb') as img:
            content = img.read()
            #we first encode to b64, which returns in byte
            #then we change it again to utf8 string type so that it can be jsonify
            output['img'] = b64encode(content).decode('utf8')
        
        return output
    elif file_typ in vid_formats:
        filename = "./exp/target_image.webm"
        output = {
            "type" : "vid",
            "prompt": "Active learning for videos is still in progress"
        }
        #send logs
        with open("./predict.log", "r", encoding='utf-8') as log:
            content = log.read()
            output['logs'] = content
        #delete the predict.py log file
        if os.path.exists("predict.log"):
            os.remove("predict.log")
        #send video
        with open(filename, 'rb') as f:
            content = f.read()
            #we first encode to b64, which returns in byte
            #then we change it again to utf8 string type so that it can be jsonify
            output['vid'] = b64encode(content).decode('utf8')
        
        return output
        
if __name__ == '__main__': 
    api.run(host='0.0.0.0', 
            debug=True, 
            port=int(os.environ.get("PORT", 8080))
           ) 
