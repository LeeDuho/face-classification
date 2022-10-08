import cv2
import os
import glob

face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')

#영문으로 폴더 이름 바꾸기
for index in range(10):
    name = girl_list[index]
    eng_name = girl_list_eng[index]
    origin_test_path = os.path.join('celeb_dataset','test',name)
    eng_test_path = os.path.join('celeb_dataset','test',eng_name)
    os.rename(origin_test_path,eng_test_path)
    origin_train_path = os.path.join('celeb_dataset','train',name)
    eng_train_path = os.path.join('celeb_dataset','train',eng_name)
    os.rename(origin_train_path,eng_train_path)
    
for name in girl_list_eng:
    cnt = 1
    fpath = os.path.join('celeb_dataset','test',name,"*.jpg")  #test파일 이름 바꾸가
    for fpath in glob.glob(fpath):
        path_r = "celeb_dataset/test/"+name+'/'+str(cnt)+".jpg"
        os.rename(fpath,path_r)
        cnt = cnt+1
        
for name in girl_list_eng:
    cnt = 1
    fpath_train = os.path.join('celeb_dataset','train',name,"*.jpg")  #train파일 이름 바꾸가
    for fpath_train in glob.glob(fpath_train):
        path_r_train = "celeb_dataset/train/"+name+'/'+str(cnt)+".jpg"
        os.rename(fpath_train,path_r_train)
        cnt = cnt+1
        
for girl_name in girl_list_eng:
    for num in range(1,800):
        try:
            print("celeb_dataset/test/"+girl_name+'/'+str(num)+".jpg")
            img = cv2.imread(os.path.join('celeb_dataset', 'test', girl_name, f'{num}.jpg'))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3,5)
            for (x,y,w,h) in faces:
                # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
                cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
                cv2.imwrite(os.path.join('celeb_dataset', 'test', girl_name, f'{num}.png'), cropped)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)
        except:
            print("error!!! : celeb_dataset/test/"+girl_name+'/'+str(num)+".jpg")
            continue