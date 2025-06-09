import json
from datetime import datetime, timedelta, timezone

with open("launch-spec.json", "r") as f:
    data = json.load(f)

now = datetime.now(timezone.utc)
in_24h = now + timedelta(hours=24)

iso_now = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
iso_24h = in_24h.strftime("%Y-%m-%dT%H:%M:%S.000Z")

data["ValidFrom"] = iso_now
data["ValidUntil"] = iso_24h

with open("launch-spec.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated ValidFrom: {iso_now}")
print(f"Updated ValidUntil: {iso_24h}")