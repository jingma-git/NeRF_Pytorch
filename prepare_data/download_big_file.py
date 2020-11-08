import os
import time
file_path = "/home/server/MaJing/Dataset/DTU.zip"
# "wget -c https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip"
while True:
    modified_time = os.path.getmtime(file_path)
    time_diff = time.time() - modified_time
    if time_diff > 10 * 60:
        print("restart down at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        os.system("wget -c https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip")
