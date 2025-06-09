import modules.aws as aws

import re
import imageio.v2 as imageio
import glob
import boto3
import datetime
import os


s3 = boto3.client('s3')
bucket_name = 'alexv2'

def extract_datetime_from_filename(filename):
    # Example filename: cube_rendering_20240610_1530.png
    match = re.search(r'_(\d{8}_\d{4})\.png$', filename)
    if match:
        dt_str = match.group(1)
        return datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M")
    return None

def download_s3_folder(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            s3_key = obj['Key']
            # Remove the prefix from the key to get the local file path
            local_path = os.path.join(local_dir, os.path.relpath(s3_key, s3_prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading {s3_key} to {local_path}")
            s3.download_file(bucket_name, s3_key, local_path)

def extract_frequency_from_filename(filename):
    # Looks for a number followed by 'Hz' in the filename
    match = re.search(r'_(\d+(?:\.\d+)?)_Hz', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def list_of_files(fromFileName, toFileName):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='sonar-render/')

    files = []
    fromTime = extract_datetime_from_filename(fromFileName)
    toTime = extract_datetime_from_filename(toFileName)
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.png'):
                # Generate a URL for each file
                print(f"https://{bucket_name}.s3.amazonaws.com/{obj['Key']}")
                
                fileTime = extract_datetime_from_filename(obj['Key'])
                if fileTime == None or fileTime < fromTime or fileTime > toTime:
                    continue
                
                freq = extract_frequency_from_filename(obj['Key'])
                if freq is None:
                    print(f"Skipping {obj['Key']} due to missing frequency information.")
                    continue
                
                if not obj['Key'].startswith('sonar-render/cube'):
                    continue
                
                file = dict()
                file['time'] = fileTime
                file['name'] = obj['Key']
                file['frequency'] = freq
                files.append(file)
                
    # After populating files list
    files_sorted = sorted(files, key=lambda f: f['frequency'])

    # Example: print sorted file names
    for f in files_sorted:
        print(f['name'])   

    return files_sorted


def make_animation(png_files):
    """
    Create an MP4 animation from a list of PNG files.
    """
    if not png_files:
        print("No PNG files found.")
        return

    # Create an MP4 animation
    with imageio.get_writer('animation.mp4', fps=10) as writer:
        for f in png_files:
            file_name = f['name']
            local_filename = os.path.basename(file_name)
            local_file_path = f'./animation/{local_filename}'            
            image = imageio.imread(local_file_path)
            writer.append_data(image)
            print(f"Added {local_file_path} to animation.")

    print("MP4 animation saved as animation.mp4")


def copy_files_from_s3(files_sorted):
    """
    Download files from S3 to local directory.
    """
    if not os.path.exists('./animation'):
        os.makedirs('./animation')

    for f in files_sorted:
        file_name = f['name']
        local_filename = os.path.basename(file_name)
        local_file_path = f'./animation/{local_filename}'
        
        # Check if the file already exists
        if os.path.exists(local_file_path):
            print(f"File already exists: {local_file_path}, skipping download.")
            continue
            
        # Download the file from S3
        s3.download_file(bucket_name, file_name, local_file_path)
        print(f"Downloaded {file_name} to {local_file_path}")

aws.list_as_url()
files_sorted = list_of_files("_20250609_1346.png", "_20250609_1458.png")
copy_files_from_s3(files_sorted)

make_animation(files_sorted)
