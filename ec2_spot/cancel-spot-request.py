import boto3

ec2 = boto3.client('ec2')

# Get all spot instance requests with state 'open' (pending)
response = ec2.describe_spot_instance_requests(
    Filters=[{'Name': 'state', 'Values': ['open']}]
)

spot_request_ids = [req['SpotInstanceRequestId'] for req in response['SpotInstanceRequests']]

if spot_request_ids:
    print(f"Cancelling {len(spot_request_ids)} pending spot request(s):")
    for req_id in spot_request_ids:
        print(f"  {req_id}")
    ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=spot_request_ids)
    print("All pending spot requests cancelled.")
else:
    print("No pending spot requests found.")