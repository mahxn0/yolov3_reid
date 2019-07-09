import sys
sys.path.append("/home/mahxn0/M_DeepLearning/ReId/ReId/GetFeature")
from GetFeature import get_feature
from yolov3 import detect
import numpy as np
import cv2
import torch
import time

if __name__ == "__main__":
    capture=cv2.VideoCapture("/home/mahxn0/M_DeepLearning/ReId/1.mp4")
    font=cv2.FONT_HERSHEY_SIMPLEX
    r=detect.Detector()
    f=get_feature.Extractor()
    reid_flag=False
    reid_thresh=0.75
    file_path="/home/mahxn0/M_DeepLearning/ReId/ReId/reid.avi"
    query_feature=torch.FloatTensor().cuda()
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    video = cv2.VideoWriter( file_path, fourcc, 25, (1920,1080) )
    #1080p detect cost 50ms 
    while True:
        if capture.isOpened():
            img=capture.read()[1]
            drawimg=img.copy()
            rect=[]
            #x1,y1,x2,y2=0,0,0,0
            if img is not None:
                result=r.detect(img)
                if len(result)>0:
                    for res in result:
                        if res[0].strip('\n')=="person":
                            x1=int(res[2])
                            y1=int(res[3])
                            x2=int(res[4])
                            y2=int(res[5])
                            cv2.rectangle(drawimg,(x1,y1),(x2,y2),(0,0,255),2)
                            #cv2.putText(drawimg,res[0],(x1,y1),font,1,(0,0,255),2)
                            rectimg=img[y1:y2,x1:x2]
                            rect.append((rectimg,x1,y1,x2,y2))
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    init_rect = cv2.selectROI('detect',img, False, False)
                    x, y, w, h = init_rect
                    reid_flag=True
                    query_img=img[y:y+h,x:x+w]
                    query_feature=f.extract_feature_cv(query_img)
                    print(query_feature.shape)
                if reid_flag:
                    if len(rect)>0:
                        #all cost 50ms
                        for res in rect:
                            gallay_feature=f.extract_feature_cv(res[0])
                            #print(gallay_feature)
                            #print("两个特征维度:",query_feature.shape,gallay_feature.shape)
                            #cost 1-5 ms
                            score=torch.mm(query_feature,gallay_feature.transpose(0,1))
                            #print(score)
                            if score>reid_thresh:
                                cv2.putText(drawimg,"yes",(res[1],res[2]),font,1,(0,0,255),2)
                            else:
                                cv2.putText(drawimg,"no",(res[1],res[2]),font,1,(0,0,255),2)
                cv2.imshow("detect",drawimg)
                video.write(drawimg)
                cv2.waitKey(1)
            else:
                print("imgdata is null")
                #break
        else:
            print("file is null")
            break            


