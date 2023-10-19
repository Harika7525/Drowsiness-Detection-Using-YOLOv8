import base64
import os.path
import sys
import yaml


from Drowsiness_Detection.exception import AppException
from Drowsiness_Detection.logger import logging



def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "ro") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise AppException(e, sys) from e


def write_yaml_file(file_path: str, content: object, reaplce: bool = False)->None:
    try:
        if reaplce:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"w") as file:
            yaml.dump(content,file)
            logging.info("Successfully write_yaml_file")
    except Exception as e:
        raise AppException(e,sys)
    



def decodeImage(imgstring,filename):
    imgdata=base64.b64decode(imgstring)
    with open("./data/"+filename,"wb") as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath,'rb') as f:
        return base64.b64encode(f.read())