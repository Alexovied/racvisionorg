import sys, glob, threading, cv2
from collections import Counter
sys.path.insert(0,"/home/user/racvisionorg"); import helmet_detector as hd
INT8="--int8" in sys.argv
for m in [a for a in sys.argv[1:] if not a.startswith("--")]:
    net,_=hd.cargar_modelo(m, int8=INT8); L=threading.Lock(); t=Counter()
    for real in ("con_casco","sin_casco"):
        for p in glob.glob(f"/mnt/user-data/uploads/rac/datasets/prueba/{real}/*"):
            img=cv2.imread(p)
            if img is None: continue
            d=sorted(hd.correr_yolo(net,L,img,2),key=lambda x:-x[0][2]*x[0][3])
            t[(real,"nada" if not d else ("casco" if d[0][1].argmax()==0 else "sin"))]+=1
    print(m, "con_casco ok", t[("con_casco","casco")], "/36  sin ok", t[("sin_casco","sin")], "/50  errores", t[("con_casco","sin")]+t[("sin_casco","casco")])
