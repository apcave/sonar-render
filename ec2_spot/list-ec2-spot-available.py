import boto3

s3 = boto3.client('s3', region_name='ap-southeast-2')

# Filter for GPU instance families (p, g, inf, trn, etc.)
gpu_families = ['p', 'g']

response = ec2.describe_instance_type_offerings(
    LocationType='availability-zone'
)

gpu_types = []
for offering in response['InstanceTypeOfferings']:
    instance_type = offering['InstanceType']
    if any(instance_type.startswith(fam) for fam in gpu_families):
        gpu_types.append(instance_type)

gpu_types = sorted(set(gpu_types))

print("GPU instance types and latest Spot prices (Linux/UNIX):")
for instance_type in gpu_types:
    price_resp = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=['Linux/UNIX'],
        MaxResults=1
    )
    if price_resp['SpotPriceHistory']:
        price = price_resp['SpotPriceHistory'][0]['SpotPrice']
        az = price_resp['SpotPriceHistory'][0]['AvailabilityZone']
        print(f"{instance_type}: ${price} in {az}")
    else:
        print(f"{instance_type}: No recent spot price data")