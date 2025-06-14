import boto3


ec2 = boto3.client('ec2', region_name='ap-southeast-2')

response = ec2.describe_instances(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)

for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"InstanceId: {instance['InstanceId']}, Type: {instance['InstanceType']}, Public IP: {instance.get('PublicIpAddress')}, State: {instance['State']['Name']}")
        