import json

# Path to the input JSON file
input_file_path = "../Resource/Dataset/Metric-Miner-Github/github_metrics.json"
# Path to the output JSON file
output_file_path = "../Resource/Dataset/Metric-Miner-Github/github_metrics_cleaned.json"

# Load the JSON data from the file
with open(input_file_path, "r") as file:
    data = json.load(file)

# Iterate through the packages and remove the specified fields
for package in data.get("packages", {}).values():
    package.pop("contributors", None)
    package.pop("contributors_unique_merged", None)

# Save the modified JSON data to the output file
with open(output_file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"Processed JSON saved to {output_file_path}")