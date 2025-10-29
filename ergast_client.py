import pandas as pd
import requests
from ratelimit import limits, sleep_and_retry
import time

# Rate limits for the Jolpica API
# 4 requests per second
ONE_SECOND = 1
MAX_CALLS_PER_SECOND = 4
# 500 requests per hour
ONE_HOUR = 3600
MAX_CALLS_PER_HOUR = 500

BASE_URL = "https://api.jolpi.ca/ergast/"

class ErgastClient:
    """
    A client for interacting with the Ergast API, including rate limiting,
    retries, and caching.
    """
    def __init__(self, retries=3, backoff_factor=0.3):
        self.session = requests.Session()
        self.cache = {}
        self.retries = retries
        self.backoff_factor = backoff_factor

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_SECOND, period=ONE_SECOND)
    @limits(calls=MAX_CALLS_PER_HOUR, period=ONE_HOUR)
    def _get(self, url, params=None):
        """
        Makes a GET request to the Ergast API with rate limiting and retries.
        Caches successful responses.
        """
        # Create a cache key from the URL and params
        cache_key = (url, tuple(sorted(params.items())) if params else None)

        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                json_response = response.json()
                self.cache[cache_key] = json_response
                return json_response
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from {url}: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor * (2 ** attempt))
                else:
                    return None

    def get_lap_times(self, season: int, round: int) -> pd.DataFrame:
        """
        Fetches lap times for a specific race from the Ergast API.
        This function handles pagination to retrieve all lap times.
        """
        all_laps_data = []
        offset = 0
        limit = 100

        while True:
            url = f"{BASE_URL}f1/{season}/{round}/laps.json"
            params = {'limit': limit, 'offset': offset}
            response = self._get(url, params=params)

            if not response:
                break

            mr_data = response.get('MRData', {})
            race_table = mr_data.get('RaceTable', {})
            races = race_table.get('Races', [])

            if not races:
                break

            laps = races[0].get('Laps', [])
            for lap in laps:
                lap_number = lap.get('number')
                for timing in lap.get('Timings', []):
                    all_laps_data.append({
                        'LapNumber': int(lap_number),
                        'driverId': timing.get('driverId'),
                        'position': int(timing.get('position')),
                        'time': timing.get('time')
                    })

            total_results = int(mr_data.get('total', 0))
            if offset + limit >= total_results:
                break

            offset += limit

        return pd.DataFrame(all_laps_data)