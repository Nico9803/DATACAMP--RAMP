import shutil
import os
from os.path import join
import gdown

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

    return