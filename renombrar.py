import os

base_dir = r"c:\Users\Alex\Desktop\ae"
folders = ["con_casco", "sin_casco"]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        continue
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # First, rename to a temporary name to avoid collision
    temp_names = []
    for i, file in enumerate(files):
        ext = os.path.splitext(file)[1].lower()
        if ext == '.jpeg': # standardize to .jpg
            ext = '.jpg'
        temp_name = f"temp_{i}{ext}"
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, temp_name))
        temp_names.append(temp_name)
        
    # Now rename to sequential numbers
    for i, temp_name in enumerate(temp_names):
        ext = os.path.splitext(temp_name)[1]
        final_name = f"{i+1}{ext}"
        os.rename(os.path.join(folder_path, temp_name), os.path.join(folder_path, final_name))
        
    print(f"Renamed {len(files)} files in {folder}")

print("Listo. Imágenes renombradas de forma secuencial.")
