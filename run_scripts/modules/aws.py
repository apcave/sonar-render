import boto3
import datetime

s3 = boto3.client('s3')
bucket_name = 'alexv2'

def copy_file_to_s3(file_name):
    """
    Copies a timestamped file to the s3 bucket.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    s3_key = 'sonar-render/' + f"{file_name}_{timestamp}.png"
    file_name += '.png'
    
    s3.upload_file(file_name, bucket_name, s3_key)

def list_as_url():
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='sonar-render/')

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.png'):
                # Generate a URL for each file
                print(f"https://{bucket_name}.s3.amazonaws.com/{obj['Key']}")

    else:
        print("No files found in the bucket.")
        
# copy_file_to_s3("cube_rendering")
# list_as_url()
        