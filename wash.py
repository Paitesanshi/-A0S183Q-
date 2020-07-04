from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt

features = ["DATE", "PRCP", "PRCP_ATTRIBUTES", "SNWD", "SNWD_ATTRIBUTES", "TAVG", "TAVG_ATTRIBUTES",
            "TMAX", "TMAX_ATTRIBUTES", "TMIN", "TMIN_ATTRIBUTES"]
DailySummary = namedtuple("DailySummary", features)
def read_weather_data(url, api_key, target_date, days):
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        if response.status_code == 200:
            data = response.json()['history']['dailysummary'][0]
            records.append(DailySummary(
                date=target_date,
                meantempm=data['meantempm'],
                meandewptm=data['meandewptm'],
                meanpressurem=data['meanpressurem'],
                maxhumidity=data['maxhumidity'],
                minhumidity=data['minhumidity'],
                maxtempm=data['maxtempm'],
                mintempm=data['mintempm'],
                maxdewptm=data['maxdewptm'],
                mindewptm=data['mindewptm'],
                maxpressurem=data['maxpressurem'],
                minpressurem=data['minpressurem'],
                precipm=data['precipm']))
        time.sleep(6)
        target_date += timedelta(days=1)
    return records

df = pd.DataFrame(records, columns=features).set_index('date')