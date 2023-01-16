import requests
import pandas as pd

# *: api(cites_to__search=slave)

# Search for cases related to slave
query = "slave"
url = f"https://api.case.law/v1/cases/?search={query}"
key = '519e093c5a6dbbbd322f9a0e931d7b12e4f9d471'

# Number of pages to fetch
pages = 10000

case_list = []
for page in range(1, pages + 1):
    # Send the request with the appropriate page number
    response = requests.get(url, params={"page": page, "decision_date_max": '1865-01-31'},
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

# Create a DataFrame from the case_list
cases_df = pd.DataFrame(case_list)
cases_df = cases_df.T

# You can print the dataframe to see the values
# print(cases_df.head)
