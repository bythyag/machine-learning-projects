import os
import json
from apify_client import ApifyClient
import json

# Set API token as an environment variable
API_TOKEN = os.getenv("APIFY_API_TOKEN")
if not API_TOKEN:
    raise ValueError("API token not found. Please set the APIFY_API_TOKEN environment variable.")

# Load cookies from the JSON file, cookies is the linkedin session cookie.
with open('cookies.json', 'r') as f:
    cookies = json.load(f)

# Initialize the ApifyClient with the API token
client = ApifyClient(API_TOKEN)

# Username-to-URL mapping
username_url_mapping = {
    "6sense": "https://www.linkedin.com/company/6sense/",
    "getwriter": "https://www.linkedin.com/company/getwriter/",
    "useinsider": "https://www.linkedin.com/company/useinsider/",
    "placer": "https://www.linkedin.com/company/placer/",
    "bluecore": "https://www.linkedin.com/company/bluecore/",
    "nearsure": "https://www.linkedin.com/company/nearsure/",
    "pixisai": "https://www.linkedin.com/company/pixisai/",
    "activecampaign": "https://www.linkedin.com/company/activecampaign/",
    "avaus": "https://www.linkedin.com/company/avaus/"
}

# Directory to save the JSON files
output_dir = "linkedin-post"
os.makedirs(output_dir, exist_ok=True)


# Iterate through the dictionary
for username, url in username_url_mapping.items():
    print(f"Processing URL for username: {username}")
    
    # Prepare the Actor input
    run_input = {
        "urls": [url],  # URL passed as a list
        "deepScrape": True,
        "rawData": False,
        "minDelay": 2,
        "maxDelay": 8,
        "limitPerSource": 100,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyCountry": "US",
        },
        "cookie": cookies,
    }

    try:
        # Run the Actor and wait for it to finish
        run = client.actor("kfiWbq3boy3dWKbiL").call(run_input=run_input)

        # Fetch Actor results from the run's dataset
        results = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        # Save results to a JSON file using the username as the filename
        output_file = os.path.join(output_dir, f"{username}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved for {username} in {output_file}")

    except Exception as e:
        print(f"Error processing {username}: {e}")
