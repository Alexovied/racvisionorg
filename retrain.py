"""
retrain.py - Reentrenamiento correcto del modelo de cascos para mineria.
IMPORTANTE: No aplica preprocess_input en el dataset - el modelo lo hace internamente.
"""
import os, sys, random
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE        = 128
BATCH_SIZE      = 16
EPOCHS_HEAD     = 30
EPOCHS_FINETUNE = 40
LR_HEAD         = 1e-3
LR_FINETUNE     = 5e-6
DATA_DIR        = Path(__file__).parent
CON_DIR         = DATA_DIR / "con_casco"
SIN_DIR         = DATA_DIR / "sin_casco"
OUTPUT_MODEL    = DATA_DIR / "modelo.keras"
OUTPUT_BACKUP   = DATA_DIR / "modelos" / "modelo_backup_pre_retrain.keras"
AUGMENT_FACTOR  = 6

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def augment_image(img_bgr):
    result = img_bgr.copy()
    # Brillo/contraste
    if random.random() < 0.8:
        alpha = random.uniform(0.5, 1.8)
        beta  = random.randint(-70, 70)
        result = np.clip(alpha * result.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    # HSV jitter (aprende por geometria, no por color)
    if random.random() < 0.8:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-20, 20)) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.2, 1.8), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.3, 1.7), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Ruido gaussiano
    if random.random() < 0.6:
        sigma = random.uniform(5, 35)
        noise = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Cutout (oclusiones)
    if random.random() < 0.5:
        h, w = result.shape[:2]
        for _ in range(2):
            hole_h = int(h * random.uniform(0.05, 0.2))
            hole_w = int(w * random.uniform(0.05, 0.2))
            y0 = random.randint(0, h - hole_h)
            x0 = random.randint(0, w - hole_w)
            result[y0:y0+hole_h, x0:x0+hole_w] = np.random.randint(0, 256, (hole_h, hole_w, 3), dtype=np.uint8)
    # Flip
    if random.random() < 0.5:
        result = cv2.flip(result, 1)
    # CLAHE
    if random.random() < 0.4:
        result = apply_clahe(result)
    return result

def load_dataset():
    X, y = [], []
    ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for label, folder in [(0, CON_DIR), (1, SIN_DIR)]:
        files = [f for f in folder.iterdir() if f.suffix.lower() in ext]
        print(f"[Dataset] {len(files)} imagenes de {folder.name} (label={label})")
        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Convertir a RGB (uint8, 0-255) - el modelo aplica preprocess internamente
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img_rgb)
            y.append(label)
            for _ in range(AUGMENT_FACTOR):
                aug = augment_image(img)
                aug = cv2.resize(aug, (IMG_SIZE, IMG_SIZE))
                X.append(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
                y.append(label)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)
    print(f"[Dataset] Total: {len(X)} ({np.sum(y==0)} con_casco | {np.sum(y==1)} sin_casco)")
    return X, y

def main():
    import keras
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input
    from keras import layers, models, regularizers, Input, optimizers, callbacks

    print("="*60)
    print("  REENTRENAMIENTO CORRECTO - CASCOS MINERIA")
    print("="*60)

    OUTPUT_BACKUP.parent.mkdir(exist_ok=True)
    if OUTPUT_MODEL.exists():
        import shutil; shutil.copy2(OUTPUT_MODEL, OUTPUT_BACKUP)
        print(f"[+] Backup guardado en {OUTPUT_BACKUP}")

    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cwd = dict(enumerate(cw))
    print(f"[+] Pesos de clase: {cwd}")

    # Convertir a float y aplicar preprocess_input
    X_train_f = preprocess_input(X_train.astype(np.float32))
    X_val_f   = preprocess_input(X_val.astype(np.float32))

    # Modelo: MobileNetV2 SIN preprocess en las capas (ya se hizo arriba)
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inp, out)

    print(f"\n{'='*60}\n  FASE 1: Clasificador ({EPOCHS_HEAD} epochs)\n{'='*60}")
    model.compile(optimizer=optimizers.Adam(LR_HEAD),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(X_train_f, y_train, validation_data=(X_val_f, y_val),
                   epochs=EPOCHS_HEAD, batch_size=BATCH_SIZE, class_weight=cwd, verbose=1,
                   callbacks=[
                       callbacks.EarlyStopping("val_accuracy", patience=10, restore_best_weights=True, verbose=1),
                       callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
                   ])

    print(f"\n{'='*60}\n  FASE 2: Fine-tuning ({EPOCHS_FINETUNE} epochs)\n{'='*60}")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(LR_FINETUNE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(X_train_f, y_train, validation_data=(X_val_f, y_val),
                   epochs=EPOCHS_FINETUNE, batch_size=BATCH_SIZE, class_weight=cwd, verbose=1,
                   callbacks=[
                       callbacks.EarlyStopping("val_accuracy", patience=15, restore_best_weights=True, verbose=1),
                       callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=5, min_lr=1e-9, verbose=1),
                       callbacks.ModelCheckpoint(str(OUTPUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1),
                   ])

    loss, acc = model.evaluate(X_val_f, y_val, verbose=0)
    print(f"\n[+] Val Accuracy: {acc*100:.1f}%")

    # Verificacion rapida de etiquetas
    print("\n[+] Verificacion de etiquetas:")
    for lbl, path in [("CON_CASCO", CON_DIR / "1.jpg"), ("SIN_CASCO", SIN_DIR / "1.jpg")]:
        img = cv2.imread(str(path))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_pre = preprocess_input(img_rgb)
        r = model(np.expand_dims(img_pre, 0), training=False).numpy()[0]
        print(f"  {lbl}: idx0(con)={r[0]*100:.1f}% idx1(sin)={r[1]*100:.1f}% => idx{np.argmax(r)}")

    if not OUTPUT_MODEL.exists():
        model.save(str(OUTPUT_MODEL))
    print("\n[OK] Listo. Reinicia el servidor: python app.py")

if __name__ == "__main__":
    main()
