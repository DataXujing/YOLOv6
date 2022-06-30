import os
import cv2

import shutil
import uuid


print(str(uuid.uuid1()).replace("-","_"))


def dat_rename(file_path, mode):

    files = os.listdir(file_path)

    for file in files:


        if ".jpg" in file:
            file_name = ".".join(file.split(".")[:-1])

            new_name = mode+"_"+str(uuid.uuid1()).replace("-","_")

            print(file)
            os.rename("./Annotations/"+file_name+".xml", "Annotations/"+new_name+".xml") 
            os.rename("./JPEGImages/"+file, "JPEGImages/"+ new_name+".jpg") 
            #normal 时
            # os.rename(file_path+"/"+file,file_path+"/"+new_name+".jpg")




if __name__ == "__main__":
    file_path = "./JPEGImages"
    mode = "2022_06_29"
    dat_rename(file_path, mode)