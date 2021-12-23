
import os
from entsoe import EntsoePandasClient

def load_data(start, end,country_code, file_path, file_name, token):
    """
    Load Data from Transparency Platform
    """

    client = EntsoePandasClient(api_key=token)

    # print(client.query_load(country_code, start=start, end=end))
    print(client.query_load_and_forecast(country_code, start=start, end=end))

    df = client.query_load_and_forecast(country_code, start=start, end=end)

    df.to_csv(os.path.join(file_path, file_name), sep = ",")

    return df