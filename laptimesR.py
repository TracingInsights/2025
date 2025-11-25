import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import pandas as pd
import requests
from joblib import Memory, Parallel, delayed

import utils
from ergast_client import ErgastClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("laptimes_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("laptimes_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

# Enable caching
fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025

# Global cache for session objects to prevent reloading
SESSION_CACHE = {}
ERGAST_LAP_CACHE = {}

class LaptimeExtractor:
    """A class to handle extraction of F1 laptime data."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
    ):
        """Initialize the LaptimeExtractor."""
        self.year = year
        self.ergast_client = ErgastClient()

        self.events = events or [
            # "Australian Grand Prix",
            # "Chinese Grand Prix",
            # "Japanese Grand Prix",
            # "Bahrain Grand Prix",
            # 'Saudi Arabian Grand Prix',
            # "Miami Grand Prix",
            # "Emilia Romagna Grand Prix",
            # "Monaco Grand Prix",
            # 'Spanish Grand Prix',
            # "Canadian Grand Prix",
            # "Austrian Grand Prix",
            # "British Grand Prix",
            # "Belgian Grand Prix",
            # "Hungarian Grand Prix",
            # "Dutch Grand Prix",
            # 'Italian Grand Prix',
            # 'Azerbaijan Grand Prix',
            # 'Singapore Grand Prix',
            # 'United States Grand Prix',
            # 'Mexico City Grand Prix',
            # 'São Paulo Grand Prix',
            'Las Vegas Grand Prix',
            # 'Qatar Grand Prix',
            # 'Abu Dhabi Grand Prix',
        ]
        self.sessions = sessions or ["Race"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached session object to prevent reloading."""
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """Get list of driver codes for a given event and session."""
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def _get_lap_times_from_ergast(
        self, event: Union[str, int], session: str, driver: str, f1session=None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches lap times for a specific driver from the Ergast API using the custom client.
        This function fetches all lap times for a race and caches them.
        """
        if session != "Race":
            return None

        cache_key = f"{self.year}-{event}-{session}"
        if cache_key in ERGAST_LAP_CACHE:
            laps_df = ERGAST_LAP_CACHE[cache_key]
            driver_laps = laps_df[laps_df["driverId"] == f1session.get_driver(driver)["DriverId"]]
            return driver_laps[["LapNumber", "LapTime_Ergast"]]

        if f1session is None:
            f1session = self.get_session(event, session)

        try:
            all_laps_df = self.ergast_client.get_lap_times(
                season=self.year, round=f1session.event["RoundNumber"]
            )

            if all_laps_df.empty:
                logger.info(f"No lap times found on Ergast for {event} {session}")
                ERGAST_LAP_CACHE[cache_key] = pd.DataFrame()
                return None

            # Ensure 'time' column is string before concatenation
            all_laps_df["time"] = all_laps_df["time"].astype(str)
            all_laps_df["LapTime_Ergast"] = pd.to_timedelta("00:" + all_laps_df["time"])

            ERGAST_LAP_CACHE[cache_key] = all_laps_df

            driver_info = f1session.get_driver(driver)
            if driver_info is None or "DriverId" not in driver_info:
                logger.warning(f"Could not find DriverId for {driver}")
                return None
            driver_id = driver_info["DriverId"]

            driver_laps = all_laps_df[all_laps_df["driverId"] == driver_id]

            return driver_laps[["LapNumber", "LapTime_Ergast"]]

        except Exception as e:
            logger.error(f"Error fetching lap times from Ergast for {driver}: {str(e)}")
            return None

    def laps_data(
        self, event: Union[str, int], session: str, driver: str, f1session=None
    ) -> Dict[str, List]:
        """Get lap data for a specific driver in a session."""
        try:
            if f1session is None:
                f1session = self.get_session(event, session)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            # Try to get lap times from Ergast and overwrite
            if session == "Race":
                ergast_laps = self._get_lap_times_from_ergast(event, session, driver, f1session)
                if ergast_laps is not None and not ergast_laps.empty:
                    ergast_lap_map = {
                        int(lap['LapNumber']): lap['LapTime_Ergast']
                        for _, lap in ergast_laps.iterrows()
                    }
                    driver_laps['LapTime'] = driver_laps['LapNumber'].map(ergast_lap_map).fillna(driver_laps['LapTime'])

            # Helper function to convert timedelta to seconds
            def timedelta_to_seconds(time_value):
                if pd.isna(time_value) or not hasattr(time_value, "total_seconds"):
                    return "None"
                return round(time_value.total_seconds(), 3)

            # Convert lap times to seconds and handle NaN values
            lap_times = [
                timedelta_to_seconds(lap_time) for lap_time in driver_laps["LapTime"]
            ]

            # Convert sector times to seconds
            sector1_times = [
                timedelta_to_seconds(s1_time) for s1_time in driver_laps["Sector1Time"]
            ]
            sector2_times = [
                timedelta_to_seconds(s2_time) for s2_time in driver_laps["Sector2Time"]
            ]
            sector3_times = [
                timedelta_to_seconds(s3_time) for s3_time in driver_laps["Sector3Time"]
            ]

            # Handle NaN values in compounds
            compounds = []
            for compound in driver_laps["Compound"]:
                if pd.isna(compound):
                    compounds.append("None")
                else:
                    compounds.append(compound)

            # Handle stint information
            stints = []
            for stint in driver_laps["Stint"]:
                if pd.isna(stint):
                    stints.append("None")
                else:
                    stints.append(int(stint))

            # Handle TyreLife
            tyre_life = []
            for life in driver_laps["TyreLife"]:
                if pd.isna(life):
                    tyre_life.append("None")
                else:
                    tyre_life.append(int(life))

            # Handle Position
            positions = []
            for pos in driver_laps["Position"]:
                if pd.isna(pos):
                    positions.append("None")
                else:
                    positions.append(int(pos))

            # Handle TrackStatus
            track_status = []
            for status in driver_laps["TrackStatus"]:
                if pd.isna(status):
                    track_status.append("None")
                else:
                    track_status.append(str(status))

            # Handle IsPersonalBest
            is_personal_best = []
            for is_pb in driver_laps["IsPersonalBest"]:
                if pd.isna(is_pb):
                    is_personal_best.append("None")
                else:
                    is_personal_best.append(bool(is_pb))

            return {
                "time": lap_times,
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": compounds,
                "stint": stints,
                "s1": sector1_times,
                "s2": sector2_times,
                "s3": sector3_times,
                "life": tyre_life,
                "pos": positions,
                "status": track_status,
                "pb": is_personal_best,
            }
        except Exception as e:
            logger.error(
                f"Error getting lap data for {driver} in {event} {session}: {str(e)}"
            )
            return {
                "time": [],
                "lap": [],
                "compound": [],
                "stint": [],
                "s1": [],
                "s2": [],
                "s3": [],
                "life": [],
                "pos": [],
                "status": [],
                "pb": [],
            }

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver."""
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            # Save lap times
            laptimes = self.laps_data(event, session, driver, f1session)
            # Replace NaN values with None before JSON serialization
            laptimes["time"] = ["None" if pd.isna(x) else x for x in laptimes["time"]]
            laptimes["lap"] = ["None" if pd.isna(x) else x for x in laptimes["lap"]]
            laptimes["compound"] = [
                "None" if pd.isna(x) else x for x in laptimes["compound"]
            ]
            with open(f"{driver_dir}/laptimes.json", "w") as json_file:
                json.dump(laptimes, json_file)

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        """Process a single event and session, extracting all laptimes data."""
        logger.info(f"Processing {event} - {session}")

        # Create base directory for this event/session
        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            # Load session data once
            f1session = self.get_session(event, session, load_telemetry=True)

            # Get driver list
            drivers = self.session_drivers_list(event, session)

            # Process drivers in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.process_driver, event, session, driver, base_dir, f1session
                    )
                    for driver in drivers
                ]

                for future in as_completed(futures):
                    future.result()  # Just to catch any exceptions

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self, max_workers: int = 4) -> None:
        """Process all configured events and sessions, with parallelization."""
        logger.info(f"Starting laptimes extraction for {self.year} season")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        start_time = time.time()

        # Process each event and session in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    futures.append(
                        executor.submit(self.process_event_session, event, session)
                    )

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Laptimes extraction completed in {elapsed_time:.2f} seconds")

def is_data_available(year, events, sessions):
    """
    Check if data is available for the specified year, events, and sessions.
    """
    try:
        # Try to load the first event and session as a test
        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data availability for {year} {event} {session}...")

        # Try to get the session without loading telemetry
        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        # Check if we have lap data
        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False

        # Check if we have at least one driver
        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    try:
        extractor = LaptimeExtractor()

        # Use more workers on GitHub Actions
        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        # Wait for data to be available
        wait_time = 30  # seconds between checks
        max_attempts = 720  # 12 hours max wait time (720 * 60 seconds)
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} season data...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info(
                    f"Data is available for {extractor.year} season. Starting extraction..."
                )
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time} seconds before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":

    main()










