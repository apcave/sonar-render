import os
import boto3
import datetime

s3 = boto3.client('s3', region_name='ap-southeast-2')
bucket_name = 'alexv2'

def copy_file_to_s3(file_name):
    """
    Copies a timestamped file to the s3 bucket.
    """
    base_file_name = os.path.basename(file_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    s3_key = 'sonar-render/' + f"{base_file_name}_{timestamp}.png"
    file_name += '.png'
    
    s3.upload_file(file_name, bucket_name, s3_key)
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        print("Upload successful!")
        print(f"https://{bucket_name}.s3.amazonaws.com/{s3_key}")
    except Exception as e:
        print("Upload failed or file not found in S3.")
        print(e)    
    

def list_as_url():
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='sonar-render/c')

    if 'Contents' in response:
        for obj in response['Contents']:
            #if obj['Key'].endswith('.png'):
                # Generate a URL for each file
            print(f"https://{bucket_name}.s3.amazonaws.com/{obj['Key']}")

    else:
        print("No files found in the bucket.")
        
# copy_file_to_s3("cube_rendering")
# list_as_url()
        