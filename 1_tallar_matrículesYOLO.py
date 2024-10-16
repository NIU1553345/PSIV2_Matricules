import torch 
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import pandas
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def tallar_coordenades(img, xmin, ymin, xmax, ymax):
    imatge = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    imatge = imatge[ymin:ymax, xmin:xmax]
    return imatge


def mat(model,i,file):       
        results=model1(i)   
        ims=np.squeeze(results.render())                
        nom, extencio = os.path.splitext(file)        
        f=nom + '_yolo' + extencio        
        ruta= os.path.join(r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\YOLO", f)    
        cv2.imwrite(ruta, ims)    
        df=results.pandas().xyxy[0]    
        df=df[df['confidence']>0.49]    
        df=df.to_dict(orient='records')
        for mat in df:   
            xmin=round(mat['xmin'])   
            ymin=round(mat['ymin'])    
            xmax=round(mat['xmax'])
            ymax=round(mat['ymax'])    
            imatge=tallar_coordenades(i, xmin, ymin, xmax, ymax)
            nou_filename = nom + '_tallada' + extencio
            path="C:/Users/Usuario/Downloads/tallades/"
            filepath_desti = os.path.join(path, nou_filename)
            cv2.imwrite(filepath_desti,imatge)
            print(f'Imatge guardada: {filepath_desti}')
            

model1=torch.hub.load('ultralytics/yolov5','custom', path='C:/Users/Usuario/OneDrive/Escriptori/UAB/4t/psiv/model10/best10.pt')#, force_reload=True)
carpeta=r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\m"
for file in os.listdir(carpeta):
    i = os.path.join(carpeta, file)
    mat(model1,i,file)
    
