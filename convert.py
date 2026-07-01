import os
import rawpy
import imageio

def convert_dng_to_jpg(directory):
    if not os.path.exists(directory): return
    files = [f for f in os.listdir(directory) if f.lower().endswith('.dng')]
    for filename in files:
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, os.path.splitext(filename)[0] + '.jpg')
        
        try:
            with rawpy.imread(old_path) as raw:
                rgb = raw.postprocess()
            imageio.imsave(new_path, rgb)
            os.remove(old_path)
            print(f"Converted {filename} to JPG and removed original.")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

base_dir = r"c:\Users\Alex\Desktop\ae"
print("Converting con_casco...")
convert_dng_to_jpg(os.path.join(base_dir, "con_casco"))
print("Converting sin_casco...")
convert_dng_to_jpg(os.path.join(base_dir, "sin_casco"))
print("Done.")
