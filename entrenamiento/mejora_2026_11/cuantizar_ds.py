"""INT8 con NNCF como cuantizar.py del repo, calibrando con fotos de entrenamiento de un dataset.
Uso: cuantizar_ds.py modelo.onnx carpeta_imagenes [n=300]"""
import glob, random, sys
sys.path.insert(0, "/home/user/racvisionorg")
import cv2, nncf, openvino as ov
import helmet_detector
mod, carpeta = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rutas = sorted(glob.glob(carpeta + "/*.jpg")); random.Random(0).shuffle(rutas); rutas = rutas[:n]
def preparar(r):
    c = helmet_detector._letterbox(cv2.imread(r))[0]
    return cv2.dnn.blobFromImage(c, 1 / 255.0, (helmet_detector.INPUT_SIZE,) * 2, swapRB=True)
int8 = nncf.quantize(ov.Core().read_model(mod), nncf.Dataset(rutas, preparar), preset=nncf.QuantizationPreset.MIXED,
                     subset_size=len(rutas), ignored_scope=nncf.IgnoredScope(types=["Multiply", "Subtract", "Sigmoid"]))
ov.save_model(int8, mod.replace(".onnx", "_int8.xml")); print("ok", mod)
