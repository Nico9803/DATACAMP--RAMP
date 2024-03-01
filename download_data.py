# import shutil
import gdown
import os
import subprocess

import argparse

CHALLENGE_NAME = 'picture_reconstruction'

def data_fetch():
    """

    Parameters
    ----------

    Returns
    -------
    None
    """
    # gdown.download_folder(url)
    # os.chdir('picture_reconstruction_dataset')
    # subprocess.call(['tar', '-zxf', 'Train.tgz'])
    # subprocess.call(['tar', '-zxf', 'Test.tgz'])
    # os.chdir('..')
    return 

def data_load():
    print("Data loading")
    url = "https://drive.google.com/drive/folders/1TqEZjRlDm14QtC1cxFRmqGtzXmO6HBTP"
    gdown.download_folder(url)
    os.chdir('data')
    subprocess.call(['tar', '-zxf', 'data.tgz'])
    os.chdir('..')
    print("Unzip completed")
    return

if __name__ == "__main__":
    
    #data_fetch()
    
    print('Done')