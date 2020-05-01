import os, glob
import time

import zipfile, concurrent
from zipfile import ZipFile
import requests
from tqdm import tqdm

# %% experimental faster way to extract zip files
def _count_file_object(f):
    # Note that this iterates on 'f'.
    # You *could* do 'return len(f.read())'
    # which would be faster but potentially memory
    # inefficient and unrealistic in terms of this
    # benchmark experiment.
    total = 0
    for line in f:
        total += len(line)
    return total


def _count_file(fn):
    with open(fn, 'rb') as f:
        return _count_file_object(f)


def unzip_member(zip_filepath, filename, dest):
    with open(zip_filepath, 'rb') as f:
        zf = zipfile.ZipFile(f)
        zf.extract(filename, dest)
    fn = os.path.join(dest, filename)
    return _count_file(fn)

# TODO: DATA RACE! other python processes try to open file at the same time.
def concurrent_find_and_extract(fn, dest=os.getcwd()):
    with open(fn, 'rb') as f:
        zf = zipfile.ZipFile(f)
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for member in zf.infolist():
                futures.append(
                    executor.submit(
                        unzip_member,
                        fn,
                        member.filename,
                        dest, ))
            total = 0
            for future in concurrent.futures.as_completed(futures):
                total += future.result()
    return total


# %%
def download_file(challenge_file, chunk):
    response = requests.head(challenge_file)
    file_lenght = int(response.headers['Content-Length'])
    print(f"file lenght is {file_lenght / (1024)}kb")

    # """ stream=True means when function returns, only the response header is downloaded, response body is not. """
    r = requests.get(url=challenge_file, stream=True)  # for large response
    with open("kagglecatsanddogs_3367a.zip", "wb") as zipfile:
        pbar = tqdm(r.iter_content(chunk_size=chunk))  # whatch the progress
        for chunk in pbar:
            # writing one chunk at a time to zip file
            if chunk:
                zipfile.write(chunk)

            # This function finds and extracts already downloaded data


# finds and downloads data
# TODO: extraction time is soo much
def all_find_and_extract(zzzipfile):
    zipstat_size = os.stat(zzzipfile).st_size
    with tqdm(zipstat_size):
        with ZipFile(zzzipfile, 'r') as zip_ref:
            zip_ref.extractall()
        zip_ref.close()


# %%
def find_and_extract_all():
    ziplist = tqdm([f for f in glob.glob("*.zip")])
    for zzzipfile in ziplist:
        # concurrent_find_and_extract(zzzipfile, dest=os.getcwd())
        all_find_and_extract(zzzipfile)


# Run download_file and find_and_extract_all together
def download_extract(challenge_file, chunk):
    print("please don't interrupt")
    download_start = time.perf_counter()
    download_file(challenge_file, chunk)
    download_stop = time.perf_counter()
    print(f"download ellepsed time: {download_stop - download_start}")

    extract_start = time.perf_counter()
    find_and_extract_all()
    extract_stop = time.perf_counter()
    print(f"download ellepsed time: {download_stop - download_start}")

# extract_start = time.perf_counter()
# doex.find_and_extract_all()
# extract_stop = time.perf_counter()
# print(f"download ellepsed time: {download_stop - download_start}")
