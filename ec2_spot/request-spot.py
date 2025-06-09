import boto3
import json
from datetime import datetime, timedelta, timezone
import sys

# Load your launch specification from JSON

# if len(sys.argv) < 2:
#     print("Usage: python request_spot.py <instance_type>")
#     sys.exit(1)

# instance_type = sys.argv[1]

with open('fleet-spec.json') as f:
    launch_spec = json.load(f)

ec2 = boto3.client('ec2')

# Update the instance type
#launch_spec["InstanceType"] = instance_type
now = datetime.now(timezone.utc)
in_24h = now + timedelta(hours=24)

iso_now = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
iso_24h = in_24h.strftime("%Y-%m-%dT%H:%M:%S.000Z")

launch_spec["ValidFrom" ] = iso_now
launch_spec["ValidUntil" ] = iso_24h

# response = ec2.request_spot_instances(
#     InstanceCount=1,
#     Type='one-time',
#     LaunchSpecification=launch_spec,
#     ValidUntil=iso_24h  # Optional: set expiration
# )

response = ec2.request_spot_fleet(
    SpotFleetRequestConfig=launch_spec,
)

print(f"Updated ValidFrom: {iso_now}")
print(f"Updated ValidUntil: {iso_24h}")
print("Spot request submitted. Response:")
print(json.dumps(response, indent=2, default=str))