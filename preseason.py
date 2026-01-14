import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import orjson
import pandas as pd
import psutil
import requests

import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preseason_telemetry.log"), logging.StreamHandler()],
)
logger = logging.getLogger("preseason_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}

SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}


class PreSeasonTelemetryExtractor:
    """Extractor for F1 Pre-Season Testing telemetry data using fastf1.get_testing_session."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        test_number: int = 1,
        session_numbers: List[int] = None,
    ):
        """Initialize the PreSeasonTelemetryExtractor.

        Args:
            year: The F1 season year
            test_number: The testing event number (usually 1, sometimes 2)
            session_numbers: List of session numbers (1, 2, 3 for Practice 1, 2, 3)
        """
        self.year = year
        self.test_number = test_number
        self.session_numbers = session_numbers or [1, 2, 3]
        self.session_name_map = {
            1: "Practice 1",
            2: "Practice 2",
            3: "Practice 3",
        }

    def get_testing_session(
        self, session_number: int, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached testing session object."""
        cache_key = f"{self.year}-testing-{self.test_number}-{session_number}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_testing_session(
                self.year, self.test_number, session_number
            )
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, session_number: int) -> List[str]:
        """Get list of driver codes for a given testing session."""
        try:
            f1session = self.get_testing_session(session_number)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(
                f"Error getting driver list for testing session {session_number}: {str(e)}"
            )
            return []

    def session_drivers(self, session_number: int) -> Dict[str, List[Dict[str, str]]]:
        """Get drivers available for a given testing session."""
        try:
            f1session = self.get_testing_session(session_number)
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
                f"Error getting drivers for testing session {session_number}: {str(e)}"
            )
            return {"drivers": []}

    def laps_data(
        self, session_number: int, driver: str, f1session=None
    ) -> Dict[str, List]:
        """Get lap data for a specific driver in a testing session."""
        try:
            if f1session is None:
                f1session = self.get_testing_session(session_number)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            def timedelta_to_seconds(time_value):
                if pd.isna(time_value) or not hasattr(time_value, "total_seconds"):
                    return "None"
                return round(time_value.total_seconds(), 3)

            lap_times = [
                timedelta_to_seconds(lap_time) for lap_time in driver_laps["LapTime"]
            ]
            sector1_times = [
                timedelta_to_seconds(s1_time) for s1_time in driver_laps["Sector1Time"]
            ]
            sector2_times = [
                timedelta_to_seconds(s2_time) for s2_time in driver_laps["Sector2Time"]
            ]
            sector3_times = [
                timedelta_to_seconds(s3_time) for s3_time in driver_laps["Sector3Time"]
            ]

            compounds = []
            for compound in driver_laps["Compound"]:
                if pd.isna(compound):
                    compounds.append("None")
                else:
                    compounds.append(compound)

            stints = []
            for stint in driver_laps["Stint"]:
                if pd.isna(stint):
                    stints.append("None")
                else:
                    stints.append(int(stint))

            tyre_life = []
            for life in driver_laps["TyreLife"]:
                if pd.isna(life):
                    tyre_life.append("None")
                else:
                    tyre_life.append(int(life))

            positions = []
            for pos in driver_laps["Position"]:
                if pd.isna(pos):
                    positions.append("None")
                else:
                    positions.append(int(pos))

            track_status = []
            for status in driver_laps["TrackStatus"]:
                if pd.isna(status):
                    track_status.append("None")
                else:
                    track_status.append(str(status))

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
                f"Error getting lap data for {driver} in testing session {session_number}: {str(e)}"
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
    def _smooth_outliers_vectorized(arr, threshold, use_abs=False):
        """Vectorized outlier smoothing using forward fill logic."""
        arr = arr.copy()
        if use_abs:
            mask = np.abs(arr[1:-1]) > threshold
        else:
            mask = arr[1:-1] > threshold
        indices = np.where(mask)[0] + 1
        for i in indices:
            arr[i] = arr[i - 1]
        return arr

    @staticmethod
    def calculate_x_acceleration(vx_array, time_array, Nax):
        """Calculate and smooth X-acceleration component."""
        dtime = np.gradient(time_array)
        ax = np.gradient(vx_array) / dtime
        ax = PreSeasonTelemetryExtractor._smooth_outliers_vectorized(
            ax, 25, use_abs=False
        )
        kernel = np.ones(Nax) / Nax
        return np.convolve(ax, kernel, mode="same")

    @staticmethod
    def calculate_y_acceleration(vx_array, x_array, y_array, dist_array, Nay):
        """Calculate and smooth Y-acceleration component."""
        dx = np.gradient(x_array)
        dy = np.gradient(y_array)
        theta = np.arctan2(dy, dx + np.finfo(float).eps)
        theta[0] = theta[1]
        theta_noDiscont = np.unwrap(theta)
        ds = np.gradient(dist_array)
        dtheta = np.gradient(theta_noDiscont)
        dtheta = PreSeasonTelemetryExtractor._smooth_outliers_vectorized(
            dtheta, 0.5, use_abs=True
        )
        C = dtheta / (ds + 0.0001)
        ay = np.square(vx_array) * C
        ay[np.abs(ay) > 150] = 0
        kernel = np.ones(Nay) / Nay
        return np.convolve(ay, kernel, mode="same")

    @staticmethod
    def calculate_z_acceleration(vx_array, x_array, z_array, dist_array, Naz):
        """Calculate and smooth Z-acceleration component."""
        dx = np.gradient(x_array)
        dz = np.gradient(z_array)
        z_theta = np.arctan2(dz, dx + np.finfo(float).eps)
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)
        ds = np.gradient(dist_array)
        z_dtheta = np.gradient(z_theta_noDiscont)
        z_dtheta = PreSeasonTelemetryExtractor._smooth_outliers_vectorized(
            z_dtheta, 0.5, use_abs=True
        )
        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx_array) * z_C
        az[np.abs(az) > 150] = 0
        kernel = np.ones(Naz) / Naz
        return np.convolve(az, kernel, mode="same")

    def accCalc(
        self, telemetry: pd.DataFrame, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """Calculate acceleration components from telemetry data."""
        vx_array = (telemetry["Speed"].values / 3.6).astype(np.float64)
        time_array = (telemetry["Time"].values / np.timedelta64(1, "s")).astype(
            np.float64
        )
        x_array = telemetry["X"].values.astype(np.float64)
        y_array = telemetry["Y"].values.astype(np.float64)
        z_array = telemetry["Z"].values.astype(np.float64)
        dist_array = telemetry["Distance"].values.astype(np.float64)

        ax_smooth = self.calculate_x_acceleration(vx_array, time_array, Nax)
        ay_smooth = self.calculate_y_acceleration(
            vx_array, x_array, y_array, dist_array, Nay
        )
        az_smooth = self.calculate_z_acceleration(
            vx_array, x_array, z_array, dist_array, Naz
        )

        telemetry = telemetry.copy()
        telemetry["Ax"] = ax_smooth
        telemetry["Ay"] = ay_smooth
        telemetry["Az"] = az_smooth
        return telemetry

    def process_single_lap_telemetry_direct(
        self, telemetry: pd.DataFrame, data_key: str
    ) -> Dict:
        """Process telemetry for a single lap."""
        acc_tel = self.accCalc(telemetry, 3, 9, 9)
        time_sec = acc_tel["Time"].dt.total_seconds().values
        drs_values = acc_tel["DRS"].values
        drs_binary = np.isin(drs_values, [10, 12, 14]).astype(np.int8)
        brake_binary = (acc_tel["Brake"].values != 0).astype(np.int8)

        return {
            "tel": {
                "time": time_sec.tolist(),
                "rpm": acc_tel["RPM"].values.tolist(),
                "speed": acc_tel["Speed"].values.tolist(),
                "gear": acc_tel["nGear"].values.tolist(),
                "throttle": acc_tel["Throttle"].values.tolist(),
                "brake": brake_binary.tolist(),
                "drs": drs_binary.tolist(),
                "distance": acc_tel["Distance"].values.tolist(),
                "rel_distance": acc_tel["RelativeDistance"].values.tolist(),
                "acc_x": acc_tel["Ax"].tolist(),
                "acc_y": acc_tel["Ay"].tolist(),
                "acc_z": acc_tel["Az"].tolist(),
                "x": acc_tel["X"].values.tolist(),
                "y": acc_tel["Y"].values.tolist(),
                "z": acc_tel["Z"].values.tolist(),
                "dataKey": data_key,
            }
        }

    def process_lap(
        self,
        session_number: int,
        driver: str,
        lap_number: int,
        driver_dir: str,
        f1session=None,
        driver_laps=None,
    ) -> bool:
        """Process a single lap for a driver."""
        file_path = f"{driver_dir}/{lap_number}_tel.json"

        if os.path.exists(file_path):
            return True

        try:
            if f1session is None:
                f1session = self.get_testing_session(
                    session_number, load_telemetry=True
                )

            if driver_laps is None:
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver).copy()
                driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
                )

            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

            if selected_lap.empty:
                logger.warning(
                    f"No data for {driver} lap {lap_number} in testing session {session_number}"
                )
                return False

            telemetry = selected_lap.get_telemetry()

            session_name = self.session_name_map.get(
                session_number, f"Session{session_number}"
            )
            data_key = (
                f"{self.year}-PreSeasonTesting-{session_name}-{driver}-{lap_number}"
            )

            telemetry_data = self.process_single_lap_telemetry_direct(
                telemetry, data_key
            )

            with open(file_path, "wb") as json_file:
                json_file.write(orjson.dumps(telemetry_data))

            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return False

    def process_lap_batch(
        self,
        session_number: int,
        driver: str,
        lap_numbers: List[int],
        driver_dir: str,
        f1session=None,
    ) -> List[bool]:
        """Process a batch of laps sequentially (more efficient for GHA 2-core runners)."""
        results = []
        for lap_num in lap_numbers:
            results.append(
                self.process_lap(session_number, driver, lap_num, driver_dir, f1session)
            )
        return results

    def get_circuit_info(self, session_number: int) -> Optional[Dict[str, List]]:
        """Get circuit corner information for the testing session."""
        cache_key = f"{self.year}-testing-{self.test_number}-{session_number}"

        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_testing_session(session_number)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            try:
                circuit_info = f1session.get_circuit_info()
                corners = circuit_info.corners
                rotation = circuit_info.rotation

                corner_info = {
                    "CornerNumber": corners["Number"].tolist(),
                    "X": corners["X"].tolist(),
                    "Y": corners["Y"].tolist(),
                    "Angle": corners["Angle"].tolist(),
                    "Distance": corners["Distance"].tolist(),
                    "Rotation": rotation,
                }
                CIRCUIT_INFO_CACHE[cache_key] = corner_info
                return corner_info
            except (AttributeError, KeyError):
                circuit_info, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_info = {
                        "CornerNumber": circuit_info["Number"].tolist(),
                        "X": circuit_info["X"].tolist(),
                        "Y": circuit_info["Y"].tolist(),
                        "Angle": circuit_info["Angle"].tolist(),
                        "Distance": (circuit_info["Distance"] / 10).tolist(),
                        "Rotation": rotation,
                    }
                    CIRCUIT_INFO_CACHE[cache_key] = corner_info
                    return corner_info

            logger.warning(
                f"Could not get corner data for testing session {session_number}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting circuit info for testing session {session_number}: {str(e)}"
            )
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """Get circuit information from the MultiViewer API."""
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

            rows = []
            for entry in data["corners"]:
                rows.append(
                    (
                        float(entry.get("trackPosition", {}).get("x", 0.0)),
                        float(entry.get("trackPosition", {}).get("y", 0.0)),
                        int(entry.get("number", 0)),
                        str(entry.get("letter", "")),
                        float(entry.get("angle", 0.0)),
                        float(entry.get("length", 0.0)),
                    )
                )

            return (
                pd.DataFrame(
                    rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                ),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None, 0.0

    def process_driver(
        self, session_number: int, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver."""
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_testing_session(
                    session_number, load_telemetry=True
                )

            laptimes = self.laps_data(session_number, driver, f1session)
            laptimes["time"] = ["None" if pd.isna(x) else x for x in laptimes["time"]]
            laptimes["lap"] = ["None" if pd.isna(x) else x for x in laptimes["lap"]]
            laptimes["compound"] = [
                "None" if pd.isna(x) else x for x in laptimes["compound"]
            ]
            with open(f"{driver_dir}/laptimes.json", "wb") as json_file:
                json_file.write(orjson.dumps(laptimes))

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()
            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            lap_numbers = driver_laps["LapNumber"].tolist()

            self.process_lap_batch(
                session_number, driver, lap_numbers, driver_dir, f1session
            )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_testing_session(self, session_number: int) -> None:
        """Process a single testing session, extracting all telemetry data."""
        session_name = self.session_name_map.get(
            session_number, f"Session{session_number}"
        )
        logger.info(f"Processing Pre-Season Testing - {session_name}")

        base_dir = f"Pre-Season Testing/{session_name}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_testing_session(session_number, load_telemetry=True)

            drivers_info = self.session_drivers(session_number)
            with open(f"{base_dir}/drivers.json", "wb") as json_file:
                json_file.write(orjson.dumps(drivers_info))

            corner_info = self.get_circuit_info(session_number)
            if corner_info:
                with open(f"{base_dir}/corners.json", "wb") as json_file:
                    json_file.write(orjson.dumps(corner_info))

            drivers = self.session_drivers_list(session_number)

            for driver in drivers:
                self.process_driver(session_number, driver, base_dir, f1session)

        except Exception as e:
            logger.error(f"Error processing testing session {session_number}: {str(e)}")

    def process_all_data(self) -> None:
        """Process all configured testing sessions sequentially (optimal for GHA 2-core)."""
        logger.info(f"Starting pre-season testing extraction for {self.year}")
        logger.info(
            f"Test number: {self.test_number}, Sessions: {self.session_numbers}"
        )

        start_time = time.time()

        for session_number in self.session_numbers:
            try:
                self.process_testing_session(session_number)
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing session {session_number}: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(
            f"Pre-season testing extraction completed in {elapsed_time:.2f} seconds"
        )
        gc.collect()


def check_memory_usage(threshold_percent=80):
    """Check if memory usage exceeds threshold and clear caches if needed."""
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
        gc.collect()

        new_memory_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(
            f"New memory usage after clearing caches: {new_memory_percent:.2f}%"
        )
        return True

    return False


def is_data_available(year, test_number, session_numbers):
    """Check if pre-season testing data is available."""
    try:
        if not session_numbers:
            logger.warning("No session numbers specified to check")
            return False

        session_number = session_numbers[0]

        logger.info(
            f"Checking data availability for {year} Pre-Season Testing, test {test_number}, session {session_number}..."
        )

        f1session = fastf1.get_testing_session(year, test_number, session_number)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info(
                f"No lap data available yet for {year} Pre-Season Testing session {session_number}"
            )
            return False

        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(
                f"No driver data available yet for {year} Pre-Season Testing session {session_number}"
            )
            return False

        logger.info(
            f"Data is available for {year} Pre-Season Testing session {session_number}"
        )
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the pre-season testing extraction script."""
    try:
        extractor = PreSeasonTelemetryExtractor(
            year=2025,
            test_number=1,
            session_numbers=[1,2,3],
        )

        wait_time = 30
        max_attempts = 720
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} pre-season testing data...")

        while attempt < max_attempts:
            if is_data_available(
                extractor.year, extractor.test_number, extractor.session_numbers
            ):
                logger.info(
                    f"Data is available for {extractor.year} pre-season testing. Starting extraction..."
                )
                extractor.process_all_data()
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

