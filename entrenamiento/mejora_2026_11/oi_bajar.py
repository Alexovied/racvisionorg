import json,os,sys,urllib.request,concurrent.futures as cf
import cv2,numpy as np
sel=json.load(open('seleccion.json'));os.makedirs('img',exist_ok=True)
def bajar(s):
    p=f"img/{s['id']}.jpg"
    if os.path.exists(p): return 1
    try:
        b=urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/{s['split']}/{s['id']}.jpg",timeout=30).read()
        im=cv2.imdecode(np.frombuffer(b,np.uint8),1)
        h,w=im.shape[:2];r=1024/max(h,w)
        if r<1: im=cv2.resize(im,(int(w*r),int(h*r)),interpolation=cv2.INTER_AREA)
        cv2.imwrite(p,im,[cv2.IMWRITE_JPEG_QUALITY,90]);return 1
    except Exception as e: return 0
with cf.ThreadPoolExecutor(16) as ex: r=list(ex.map(bajar,sel))
print(sum(r),len(r))
