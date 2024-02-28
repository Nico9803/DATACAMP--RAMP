import shutil
import os
from os.path import join
# import wget
import gdown

# import hashlib
# import osfclient
# import argparse
# import tempfile
# import tarfile


def dummy_fetch(*args, **kwargs):
    """
    Doesn't fetch data; just places dummy (blank) data into datas/ directory so that it can pass unit tests. 
    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    None
    """
    # dest_path = "datas"
    # # pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
    # self_dir = os.path.dirname(__file__)

    # shutil.copytree(join(self_dir, "tests"), join(dest_path))
    return


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
    print(url)
    return