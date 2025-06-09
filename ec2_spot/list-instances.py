import boto3
import os
import re
import subprocess

def add_or_replace_ssh_config_entry(host, hostname, user, identity_file=None):
    config_path = os.path.expanduser("~/.ssh/config")
    entry = f"\nHost {host}\n    HostName {hostname}\n    User {user}\n"
    if identity_file:
        entry += f"    IdentityFile {identity_file}\n"

    # Read existing config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = f.read()
    else:
        config = ""

    # Regex to find existing Host block with the same HostName
    pattern = re.compile(
        r"(Host\s+\S+[\s\S]*?HostName\s+" + re.escape(hostname) + r"[\s\S]*?)(?=^Host\s|\Z)", 
        re.MULTILINE
    )

    # Remove existing entry with the same HostName
    config_new = re.sub(pattern, "", config)

    # Append new entry
    config_new = config_new.rstrip() 
    if len(config_new) > 0 and not config_new.endswith("\n"):
        config_new += "\n"
    config_new += entry

    # Write back
    with open(config_path, "w") as f:
        f.write(config_new)

def scp_to_ec2(local_file, remote_host, remote_path, identity_file):
    """
    Copy a file to a remote EC2 instance using scp.
    """
    cmd = [
        "scp",
        "-i", identity_file,
        local_file,
        f"ubuntu@{remote_host}:{remote_path}"
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("SCP failed:", result.stderr)
    else:
        print("SCP succeeded.")
        
def ssh_execute_command_with_alias(ssh_alias, command):
    ssh_cmd = [
        "ssh",
        ssh_alias,
        command
    ]
    print("Running:", " ".join(ssh_cmd))
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("SSH command failed:", result.stderr)
    else:
        print("SSH command output:", result.stdout)
    return result




ec2 = boto3.client('ec2')

response = ec2.describe_instances(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)

for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"InstanceId: {instance['InstanceId']}, Type: {instance['InstanceType']}, Public IP: {instance.get('PublicIpAddress')}, State: {instance['State']['Name']}")
        
        if instance['InstanceType'] != 't2.micro':
            print('Adding SSH config entry for:', instance['InstanceId'])
            pem_file = os.path.expanduser("~/Coding/AlexSpotswood.pem")
            hostname = instance.get('PublicIpAddress')
            host = 'awsSpots'
            add_or_replace_ssh_config_entry(
                host=host,
                hostname=hostname,
                user='ubuntu',
                identity_file=pem_file)
            
            scp_to_ec2('./ec2_spot.sh', hostname, '~/ec2_spot.sh', pem_file)
            scp_to_ec2(os.path.expanduser("~/.aws/config"), hostname, 'config', pem_file)
            scp_to_ec2(os.path.expanduser("~/.aws/credentials"), hostname, 'credentials', pem_file)
            ssh_execute_command_with_alias("awsSpots", "./ec2_spot.sh")