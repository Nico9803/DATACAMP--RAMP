# import shutil
import gdown
import os
import subprocess
def data_fetch():
    """

    Parameters
    ----------

    Returns
    -------
    None
    """
    
    url = "https://drive.google.com/drive/folders/1rwiMe6sSarsf51fMp_Rh_hgeTOI5WzVn"
    gdown.download_folder(url)
    os.chdir('picture_reconstruction_dataset')
    subprocess.call(['tar', '-zxf', 'Train.tgz'])
    subprocess.call(['tar', '-zxf', 'Test.tgz'])
    os.chdir('..')
    return