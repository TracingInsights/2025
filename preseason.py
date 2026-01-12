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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telemetry_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("telemetry_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

# Enable caching
fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": f"FastF1/"}

# Global cache for session objects to prevent reloading
SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}

# Initialize joblib memory for persistent caching
memory = Memory(location="./cache_joblib", verbose=0)


# Session name to number mapping for testing sessions
SESSION_NAME_TO_NUMBER = {
    "Practice 1": 1,
    "Practice 2": 2,
    "Practice 3": 3,
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
}


class TelemetryExtractor:
    """Optimized class to handle extraction of F1 pre-season testing telemetry data."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        test_number: int = 1,
        sessions: List[str] = None,
        use_joblib: bool = True,
        n_jobs: int = -1,
        batch_size: int = 8,
    ):
        """Initialize the TelemetryExtractor for pre-season testing.

        Args:
            year: The F1 season year
            test_number: The testing event number (usually 1 for pre-season)
            sessions: List of session names (e.g., ['Practice 1', 'Practice 2', 'Practice 3'])
            use_joblib: Whether to use joblib for parallel processing
            n_jobs: Number of parallel jobs (-1 for all cores)
            batch_size: Laps per batch for joblib processing
        """
        self.year = year
        self.test_number = test_number
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.batch_size = batch_size

        # Pre-season testing typically has 3 sessions (3 days of testing)
        self.sessions = sessions or [
            "Practice 1",
            "Practice 2", 
            "Practice 3",
        ]

    def _get_session_number(self, session: str) -> int:
        """Convert session name to session number."""
        if isinstance(session, int):
            return session
        return SESSION_NAME_TO_NUMBER.get(
            session, int(session) if session.isdigit() else 1
        )

    def get_session(
        self, session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached testing session object to prevent reloading."""
        session_number = self._get_session_number(session)
        cache_key = f"{self.year}-test{self.test_number}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_testing_session(
                self.year, self.test_number, session_number
            )
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, session: str) -> List[str]:
        """Get list of driver codes for a given testing session."""
        try:
            f1session = self.get_session(session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(
                f"Error getting driver list for test {self.test_number} {session}: {str(e)}"
            )
            return []

    def session_drivers(self, session: str) -> Dict[str, List[Dict[str, str]]]:
        """Get drivers available for a given testing session."""
        try:
            f1session = self.get_session(session)
            laps = f1session.laps
            team_colors = utils.team_colors(self.year)
            laps["color"] = laps["Team"].map(team_colors)

            unique_drivers = laps["Driver"].unique()

            drivers = [
                {
                    "driver": driver,
                    "team": laps[laps.Driver == driver].Team.iloc[0],
                }
                for driver in unique_drivers
            ]

            return {"drivers": drivers}
        except Exception as e:
            logger.error(
                f"Error getting drivers for test {self.test_number} {session}: {str(e)}"
            )
            return {"drivers": []}

    def laps_data(self, session: str, driver: str, f1session=None) -> Dict[str, List]:
        """Get lap data for a specific driver in a testing session."""
        try:
            if f1session is None:
                f1session = self.get_session(session)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

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

            # Handle Position - Note: Position may not be meaningful in testing
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
                f"Error getting lap data for {driver} in test {self.test_number} {session}: {str(e)}"
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

    @staticmethod
    @memory.cache
    def calculate_x_acceleration(vx_array, time_array, Nax):
        """Calculate and smooth X-acceleration component using joblib caching."""
        dtime = np.gradient(time_array)
        ax = np.gradient(vx_array) / dtime
        ax[np.isnan(ax) | np.isinf(ax)] = 0
        return np.convolve(ax, np.ones(Nax) / Nax, mode="same")

    @staticmethod
    @memory.cache
    def calculate_y_acceleration(vy_array, time_array, Nay):
        """Calculate and smooth Y-acceleration component using joblib caching."""
        dtime = np.gradient(time_array)
        ay = np.gradient(vy_array) / dtime
        ay[np.isnan(ay) | np.isinf(ay)] = 0
        return np.convolve(ay, np.ones(Nay) / Nay, mode="same")

    @staticmethod
    @memory.cache
    def apply_savgol_smoothing(data_array, window_length, poly_order):
        """Apply Savitzky-Golay filter using joblib caching."""
        from scipy.signal import savgol_filter

        if len(data_array) < window_length:
            return data_array
        return savgol_filter(
            data_array, window_length=window_length, polyorder=poly_order
        )

    def process_lap_batch(
        self,
        session: str,
        driver: str,
        lap_numbers: List[int],
        driver_dir: str,
        f1session,
        driver_laps,
    ) -> List[Dict]:
        """Process a batch of laps using joblib for improved performance."""
        results = []
        for lap_number in lap_numbers:
            try:
                result = self.process_single_lap(
                    session, driver, lap_number, driver_dir, f1session, driver_laps
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error processing lap {lap_number} for {driver}: {str(e)}"
                )
        return results

    def process_single_lap(
        self,
        session: str,
        driver: str,
        lap_number: int,
        driver_dir: str,
        f1session,
        driver_laps,
    ) -> Dict:
        """Process a single lap and extract telemetry data."""
        try:
            lap = driver_laps[driver_laps["LapNumber"] == lap_number].iloc[0]
            telemetry = lap.get_telemetry()

            if telemetry.empty:
                return {"lap": lap_number, "status": "no_telemetry"}

            # Helper function to convert array values, replacing NaN with "None"
            def to_list_with_none(arr, as_int=False):
                result = []
                for val in arr:
                    if pd.isna(val):
                        result.append("None")
                    elif as_int:
                        result.append(int(val))
                    else:
                        result.append(float(val))
                return result

            # Extract and process telemetry
            time_array = telemetry["Time"].dt.total_seconds().values
            distance = telemetry["Distance"].values
            speed = telemetry["Speed"].values
            throttle = telemetry["Throttle"].values
            brake = telemetry["Brake"].values
            gear = telemetry["nGear"].values
            rpm = telemetry["RPM"].values
            drs = telemetry["DRS"].values

            # Get position data
            x = telemetry["X"].values
            y = telemetry["Y"].values

            # Calculate velocities
            vx = np.gradient(x) / np.gradient(time_array)
            vy = np.gradient(y) / np.gradient(time_array)

            # Handle NaN/Inf values
            vx[np.isnan(vx) | np.isinf(vx)] = 0
            vy[np.isnan(vy) | np.isinf(vy)] = 0

            # Calculate accelerations with smoothing
            Nax, Nay = 5, 5
            ax_smooth = self.calculate_x_acceleration(tuple(vx), tuple(time_array), Nax)
            ay_smooth = self.calculate_y_acceleration(tuple(vy), tuple(time_array), Nay)

            # Prepare lap data with NaN handling
            lap_data = {
                "time": to_list_with_none(time_array),
                "distance": to_list_with_none(distance),
                "speed": to_list_with_none(speed),
                "throttle": to_list_with_none(throttle),
                "brake": to_list_with_none(brake),
                "gear": to_list_with_none(gear, as_int=True),
                "rpm": to_list_with_none(rpm, as_int=True),
                "drs": to_list_with_none(drs, as_int=True),
                "x": to_list_with_none(x),
                "y": to_list_with_none(y),
                "ax": to_list_with_none(ax_smooth),
                "ay": to_list_with_none(ay_smooth),
            }

            # Save lap data
            lap_file = os.path.join(driver_dir, f"{int(lap_number)}_tel.json")
            with open(lap_file, "w") as f:
                json.dump(lap_data, f)

            return {"lap": lap_number, "status": "success"}

        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return {"lap": lap_number, "status": "error", "error": str(e)}

    def get_circuit_info(self, session: str) -> Optional[Dict]:
        """Get circuit corner information for a testing session."""
        cache_key = f"{self.year}-test{self.test_number}-{session}-circuit"
        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_session(session)
            circuit_info = f1session.get_circuit_info()

            corners = []
            for idx, corner in circuit_info.corners.iterrows():
                corners.append(
                    {
                        "number": int(corner["Number"]),
                        "letter": corner["Letter"]
                        if pd.notna(corner["Letter"])
                        else "",
                        "x": float(corner["X"]),
                        "y": float(corner["Y"]),
                        "angle": float(corner["Angle"])
                        if pd.notna(corner["Angle"])
                        else 0,
                        "distance": float(corner["Distance"]),
                    }
                )

            result = {"corners": corners}
            CIRCUIT_INFO_CACHE[cache_key] = result
            return result

        except Exception as e:
            logger.error(
                f"Error getting circuit info for test {self.test_number} {session}: {str(e)}"
            )
            return None

    def process_driver(
        self,
        session: str,
        driver: str,
        base_dir: str,
        f1session,
    ) -> None:
        """Process all laps for a single driver in a testing session."""
        try:
            driver_dir = os.path.join(base_dir, driver)
            os.makedirs(driver_dir, exist_ok=True)

            # Get driver laps
            driver_laps = f1session.laps.pick_drivers(driver)

            if driver_laps.empty:
                logger.warning(
                    f"No laps found for {driver} in test {self.test_number} {session}"
                )
                return

            # Save lap summary data
            laps_info = self.laps_data(session, driver, f1session)
            with open(os.path.join(driver_dir, "laptimes.json"), "w") as f:
                json.dump(laps_info, f)

            # Get lap numbers to process
            lap_numbers = driver_laps["LapNumber"].unique().tolist()

            if self.use_joblib:
                # Process laps in batches using joblib
                batches = [
                    lap_numbers[i : i + self.batch_size]
                    for i in range(0, len(lap_numbers), self.batch_size)
                ]

                Parallel(n_jobs=self.n_jobs)(
                    delayed(self.process_lap_batch)(
                        session,
                        driver,
                        batch,
                        driver_dir,
                        f1session,
                        driver_laps,
                    )
                    for batch in batches
                )
            else:
                # Process laps sequentially with threading
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(
                            self.process_single_lap,
                            session,
                            driver,
                            lap_number,
                            driver_dir,
                            f1session,
                            driver_laps,
                        )
                        for lap_number in lap_numbers
                    ]

                    for future in as_completed(futures):
                        future.result()

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_session(self, session: str) -> None:
        """Process a single testing session, extracting all telemetry data."""
        logger.info(
            f"Processing Pre-Season Test {self.test_number} - {session} "
            f"{'with joblib' if self.use_joblib else 'without joblib'}"
        )

        # Create base directory for this test/session
        base_dir = f"Pre-Season Testing/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            # Load session data once
            f1session = self.get_session(session, load_telemetry=True)

            # Save drivers information
            drivers_info = self.session_drivers(session)
            with open(f"{base_dir}/drivers.json", "w") as json_file:
                json.dump(drivers_info, json_file)

            # Save circuit corner information
            corner_info = self.get_circuit_info(session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "w") as json_file:
                    json.dump(corner_info, json_file)

            # Get driver list
            drivers = self.session_drivers_list(session)

            # Process drivers in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.process_driver, session, driver, base_dir, f1session
                    )
                    for driver in drivers
                ]

                for future in as_completed(futures):
                    future.result()

        except Exception as e:
            logger.error(
                f"Error processing test {self.test_number} {session}: {str(e)}"
            )

    def process_all_data(self, max_workers: int = 4) -> None:
        """Process all configured testing sessions, with parallelization."""
        logger.info(
            f"Starting {'joblib-optimized' if self.use_joblib else 'standard'} telemetry extraction "
            f"for {self.year} pre-season testing"
        )
        logger.info(f"Test number: {self.test_number}")
        logger.info(f"Sessions: {self.sessions}")

        if self.use_joblib:
            logger.info(
                f"Joblib settings: n_jobs={self.n_jobs}, batch_size={self.batch_size}"
            )

        start_time = time.time()

        # Process each session in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for session in self.sessions:
                futures.append(executor.submit(self.process_session, session))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Telemetry extraction completed in {elapsed_time:.2f} seconds")

    def clear_joblib_cache(self):
        """Clear the joblib memory cache."""
        if hasattr(memory, "clear"):
            memory.clear()
            logger.info("Joblib cache cleared")


import gc
import logging
import os

import psutil

logger = logging.getLogger("memory_monitor")


def check_memory_usage(threshold_percent=80):
    """
    Check if memory usage exceeds threshold and clear caches if needed.

    Args:
        threshold_percent: Memory usage percentage threshold

    Returns:
        True if memory was cleared, False otherwise
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Current memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(
            f"Memory usage exceeds {threshold_percent}% threshold, clearing caches"
        )
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()

        if hasattr(memory, "clear"):
            memory.clear()
            logger.info("Joblib cache cleared")

        gc.collect()

        new_memory_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(
            f"New memory usage after clearing caches: {new_memory_percent:.2f}%"
        )
        return True

    return False


def is_data_available(year: int, test_number: int, sessions: List[str]) -> bool:
    """
    Check if pre-season testing data is available.

    Args:
        year: The F1 season year
        test_number: The testing event number
        sessions: List of session names to check (e.g., ['Practice 1'])

    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        if not sessions:
            logger.warning("No sessions specified to check")
            return False

        session = sessions[0]
        session_number = SESSION_NAME_TO_NUMBER.get(session, 1)

        logger.info(
            f"Checking data availability for {year} Pre-Season Test {test_number} {session}..."
        )

        # Try to get the testing session
        f1session = fastf1.get_testing_session(year, test_number, session_number)
        f1session.load(telemetry=False, weather=False, messages=False)

        # Check if we have lap data
        if f1session.laps.empty:
            logger.info(
                f"No lap data available yet for {year} Pre-Season Test {test_number} {session}"
            )
            return False

        # Check if we have at least one driver
        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(
                f"No driver data available yet for {year} Pre-Season Test {test_number} {session}"
            )
            return False

        logger.info(
            f"Data is available for {year} Pre-Season Test {test_number} {session}"
        )
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the script with joblib optimization options."""
    try:
        # Configuration options for pre-season testing
        year = 2025
        test_number = 1  # Pre-season test number (usually 1)
        sessions = ["Practice 1", "Practice 2", "Practice 3"]  # 3 days of testing

        # Joblib configuration
        use_joblib = True
        n_jobs = -1  # -1 uses all available cores
        batch_size = 8

        # Create extractor for pre-season testing
        extractor = TelemetryExtractor(
            year=year,
            test_number=test_number,
            sessions=sessions,
            use_joblib=use_joblib,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

        # Use more workers on GitHub Actions
        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        # Wait for data to be available
        wait_time = 30  # seconds between checks
        max_attempts = 720  # 12 hours max wait time
        attempt = 0

        logger.info(f"Starting to wait for {year} pre-season testing data...")

        while attempt < max_attempts:
            if is_data_available(year, test_number, sessions):
                logger.info(
                    f"Data is available for {year} pre-season testing. Starting extraction..."
                )
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time} seconds before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)

                check_memory_usage()

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
