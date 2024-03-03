# import shutil
import gdown
import os
import subprocess

import argparse

CHALLENGE_NAME = 'picture_reconstruction'

def data_load():
    print("Data loading")
    # url = "https://drive.google.com/drive/folders/1TqEZjRlDm14QtC1cxFRmqGtzXmO6HBTP" # resolution 320x320 for y (64x64 for x)
    url = "https://drive.google.com/drive/folders/1Ih5l6HYhwsBK1GYJIyAPI-qIQfz15t2B" # resolution 128x128 for y (64x64 for x)
    gdown.download_folder(url)

    # os.chdir('data')
    subprocess.call(['tar', '-zxf', 'data/data.tgz'])
    # os.chdir('..')
    
    print("Unzip completed")
    return

if __name__ == "__main__":
    
    data_load()
    
    print('Done')
