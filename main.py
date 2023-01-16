import requests
import pandas as pd

# Search API endpoint information
url = "https://api.case.law/v1/cases/?"
key = 'Token 519e093c5a6dbbbd322f9a0e931d7b12e4f9d471'

# Number of pages to fetch
pages = 10,000

case_list = []
for page in range(1, pages + 1):
    # Send the request with the appropriate page number
    response = requests.get(url, params={"page": page, "decision_date_min": "2000-01-01", "cites_to__search": "slave",
                                         "cites_to__decision_date_max": "1865-01-31", "full_case": "false",
                                         "body_format": "text"
                                         },
                            headers={'Authorization': key})

    # Check that the request was successful
    if response.status_code == 200:
        data = response.json()
        case_list += data['results']
        # Check for the last page
        if data["next"] is None:
            break
    else:
        print(f"Request failed with status code {response.status_code}")

    print(f"Page {page} Complete")

# Create a DataFrame from the case_list
cases_df = pd.DataFrame(case_list)
cases_df.drop_duplicates(subset='id', keep='first')

new_df = pd.DataFrame(cases_df['decision_date'])
new_df = new_df.sort_values(by='decision_date')
print(new_df.head())

new_df.to_csv("slavery_citing_cases.csv", index=False)
