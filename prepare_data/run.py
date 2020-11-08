# https://unix.stackexchange.com/questions/238180/execute-shell-commands-in-python
import glob
import os
# 02958343: car
# 03001627: chair

data_dir = "/home/server/MaJing/PycharmProj/nerf/data/nerf_synthetic/easytoy/"
files = glob.glob(data_dir+"*/*.obj", recursive=True)

for obj_file in files:
    file_name = os.path.basename(obj_file)
    out_path = obj_file.replace(file_name, "rendering")
    print(obj_file, file_name)
    os.system(
        "/home/server/Software/blender2.79/blender --background --python ./nerf_batch.py -- "
        "--output_folder {} {} "
        "--elevation {} "
        "--views {} "
        "--depth_scale {}".format(out_path, obj_file,
                                  30.0,
                                  100,
                                  1.4),
    )
