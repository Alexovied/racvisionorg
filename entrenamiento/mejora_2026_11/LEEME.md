# Mejora de casco, chaleco y caídas (26/11/2026)

Scripts usados para entrenar y medir los modelos de este PR. Están tal cual se corrieron (rutas de
`/home/user/work`), sirven de referencia para repetir el proceso.

Datos (todos con licencia que permite uso comercial, citando la fuente):
- **Construction-PPE** de Ultralytics, versión nov-2026: CC BY 4.0. IDs de clase: 0 helmet, 2 vest, 5 Person, 6 no_helmet, 11 no_vest.
- **Open Images V7** (val + test): anotaciones CC BY 4.0, imágenes CC BY 2.0. Cabezas, gorras y sombreros = `sin_casco`.
- **GMDCSA24** (github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos, MIT): 160 videos para medir caídas.
- Fotos propias de RAC (`con_casco/`, `sin_casco/`, `personas/`).

Pasos:
1. `oi_select.py` + `oi_bajar.py`: elegir y bajar fotos de Open Images.
2. `armar_casco.py` / `armar_chaleco.py`: armar los datasets YOLO con pruebas separadas que no se usan para elegir.
3. `entrenar.py base.pt data.yaml nombre epocas [clave=valor]`: casco desde el modelo "cerca" (15 épocas + 6 con más peso
   a fotos de RAC); chaleco desde `yolo11n.pt` (25 épocas). CPU de 4 núcleos: ~10 min por época.
4. `cuantizar_ds.py modelo.onnx carpeta_fotos`: INT8 con NNCF.
5. `eval_casco.py`, `eval_chaleco.py`, `prueba_alex.py`: miden con el mismo código de producción.
6. Caídas: `volcar_rot.py` guarda la pose de cada cuadro (derecha y girada 90°) y `reglas2.py` / `eval_caidas.py` prueban reglas.

Los pesos de YOLO (Ultralytics) son AGPL-3.0, igual que los modelos anteriores.
