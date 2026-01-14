import gc
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
import psutil
import requests
from joblib import Memory, Parallel, delayed

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

memory = Memory(location="./cache_joblib", verbose=0)


class PreSeasonTelemetryExtractor:
    """Extractor for F1 Pre-Season Testing telemetry data using fastf1.get_testing_session."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        test_number: int = 1,
        session_numbers: List[int] = None,
        use_joblib: bool = True,
        n_jobs: int = -1,
        batch_size: int = 8,
    ):
        """Initialize the PreSeasonTelemetryExtractor.

        Args:
            year: The F1 season year
            test_number: The testing event number (usually 1, sometimes 2)
            session_numbers: List of session numbers (1, 2, 3 for Practice 1, 2, 3)
            use_joblib: Whether to use joblib for parallel processing
            n_jobs: Number of jobs for joblib (-1 uses all cores)
            batch_size: Laps per batch for joblib processing
        """
        self.year = year
        self.test_number = test_number
        self.session_numbers = session_numbers or [1, 2, 3]
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.batch_size = batch_size
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
    @memory.cache
    def calculate_x_acceleration(vx_array, time_array, Nax):
        """Calculate and smooth X-acceleration component using joblib caching."""
        dtime = np.gradient(time_array)
        ax = np.gradient(vx_array) / dtime

        for i in np.arange(1, len(ax) - 1).astype(int):
            if ax[i] > 25:
                ax[i] = ax[i - 1]

        ax_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")
        return ax_smooth

    @staticmethod
    @memory.cache
    def calculate_y_acceleration(vx_array, x_array, y_array, dist_array, Nay):
        """Calculate and smooth Y-acceleration component using joblib caching."""
        dx = np.gradient(x_array)
        dy = np.gradient(y_array)

        theta = np.arctan2(dy, (dx + np.finfo(float).eps))
        theta[0] = theta[1]
        theta_noDiscont = np.unwrap(theta)

        ds = np.gradient(dist_array)
        dtheta = np.gradient(theta_noDiscont)

        for i in np.arange(1, len(dtheta) - 1).astype(int):
            if abs(dtheta[i]) > 0.5:
                dtheta[i] = dtheta[i - 1]

        C = dtheta / (ds + 0.0001)
        ay = np.square(vx_array) * C

        indexProblems = np.abs(ay) > 150
        ay[indexProblems] = 0

        ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")
        return ay_smooth

    @staticmethod
    @memory.cache
    def calculate_z_acceleration(vx_array, x_array, z_array, dist_array, Naz):
        """Calculate and smooth Z-acceleration component using joblib caching."""
        dx = np.gradient(x_array)
        dz = np.gradient(z_array)

        z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        ds = np.gradient(dist_array)
        z_dtheta = np.gradient(z_theta_noDiscont)

        for i in np.arange(1, len(z_dtheta) - 1).astype(int):
            if abs(z_dtheta[i]) > 0.5:
                z_dtheta[i] = z_dtheta[i - 1]

        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx_array) * z_C

        indexProblems = np.abs(az) > 150
        az[indexProblems] = 0

        az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")
        return az_smooth

    def accCalc(
        self, telemetry: pd.DataFrame, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """Calculate acceleration components from telemetry data."""
        vx = telemetry["Speed"] / 3.6
        time_float = telemetry["Time"] / np.timedelta64(1, "s")

        vx_array = vx.values
        time_array = time_float.values
        x_array = telemetry["X"].values
        y_array = telemetry["Y"].values
        z_array = telemetry["Z"].values
        dist_array = telemetry["Distance"].values

        if self.use_joblib and len(telemetry) > 100:
            results = Parallel(
                n_jobs=min(3, self.n_jobs if self.n_jobs > 0 else 3),
                backend="threading",
            )(
                [
                    delayed(self.calculate_x_acceleration)(vx_array, time_array, Nax),
                    delayed(self.calculate_y_acceleration)(
                        vx_array, x_array, y_array, dist_array, Nay
                    ),
                    delayed(self.calculate_z_acceleration)(
                        vx_array, x_array, z_array, dist_array, Naz
                    ),
                ]
            )

            ax_smooth, ay_smooth, az_smooth = results
        else:
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

    @staticmethod
    @memory.cache
    def cached_telemetry_processing(
        time_array,
        rpm_array,
        speed_array,
        gear_array,
        throttle_array,
        brake_array,
        drs_array,
        distance_array,
        rel_distance_array,
        x_array,
        y_array,
        z_array,
        ax_array,
        ay_array,
        az_array,
        data_key,
    ):
        """Cache the final telemetry data structure."""
        return {
            "tel": {
                "time": time_array.tolist()
                if hasattr(time_array, "tolist")
                else time_array,
                "rpm": rpm_array.tolist()
                if hasattr(rpm_array, "tolist")
                else rpm_array,
                "speed": speed_array.tolist()
                if hasattr(speed_array, "tolist")
                else speed_array,
                "gear": gear_array.tolist()
                if hasattr(gear_array, "tolist")
                else gear_array,
                "throttle": throttle_array.tolist()
                if hasattr(throttle_array, "tolist")
                else throttle_array,
                "brake": brake_array.tolist()
                if hasattr(brake_array, "tolist")
                else brake_array,
                "drs": drs_array.tolist()
                if hasattr(drs_array, "tolist")
                else drs_array,
                "distance": distance_array.tolist()
                if hasattr(distance_array, "tolist")
                else distance_array,
                "rel_distance": rel_distance_array.tolist()
                if hasattr(rel_distance_array, "tolist")
                else rel_distance_array,
                "acc_x": ax_array.tolist() if hasattr(ax_array, "tolist") else ax_array,
                "acc_y": ay_array.tolist() if hasattr(ay_array, "tolist") else ay_array,
                "acc_z": az_array.tolist() if hasattr(az_array, "tolist") else az_array,
                "x": x_array.tolist() if hasattr(x_array, "tolist") else x_array,
                "y": y_array.tolist() if hasattr(y_array, "tolist") else y_array,
                "z": z_array.tolist() if hasattr(z_array, "tolist") else z_array,
                "dataKey": data_key,
            }
        }

    def process_single_lap_telemetry_direct(
        self, telemetry: pd.DataFrame, data_key: str
    ) -> Dict:
        """Process telemetry for a single lap."""
        acc_tel = self.accCalc(telemetry, 3, 9, 9)
        acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

        acc_tel["DRS"] = acc_tel["DRS"].apply(lambda x: 1 if x in [10, 12, 14] else 0)
        acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x else 0)

        if self.use_joblib:
            return self.cached_telemetry_processing(
                acc_tel["Time"].values,
                acc_tel["RPM"].values,
                acc_tel["Speed"].values,
                acc_tel["nGear"].values,
                acc_tel["Throttle"].values,
                acc_tel["Brake"].values,
                acc_tel["DRS"].values,
                acc_tel["Distance"].values,
                acc_tel["RelativeDistance"].values,
                acc_tel["X"].values,
                acc_tel["Y"].values,
                acc_tel["Z"].values,
                acc_tel["Ax"].values,
                acc_tel["Ay"].values,
                acc_tel["Az"].values,
                data_key,
            )
        else:
            return {
                "tel": {
                    "time": acc_tel["Time"].tolist(),
                    "rpm": acc_tel["RPM"].tolist(),
                    "speed": acc_tel["Speed"].tolist(),
                    "gear": acc_tel["nGear"].tolist(),
                    "throttle": acc_tel["Throttle"].tolist(),
                    "brake": acc_tel["Brake"].tolist(),
                    "drs": acc_tel["DRS"].tolist(),
                    "distance": acc_tel["Distance"].tolist(),
                    "rel_distance": acc_tel["RelativeDistance"].tolist(),
                    "acc_x": acc_tel["Ax"].tolist(),
                    "acc_y": acc_tel["Ay"].tolist(),
                    "acc_z": acc_tel["Az"].tolist(),
                    "x": acc_tel["X"].tolist(),
                    "y": acc_tel["Y"].tolist(),
                    "z": acc_tel["Z"].tolist(),
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

            with open(file_path, "w") as json_file:
                json.dump(telemetry_data, json_file)

            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return False

    def process_lap_batch_with_joblib(
        self,
        session_number: int,
        driver: str,
        lap_numbers: List[int],
        driver_dir: str,
        f1session=None,
    ) -> List[bool]:
        """Process a batch of laps using joblib."""

        def process_single_lap_job(lap_number):
            return self.process_lap(
                session_number, driver, lap_number, driver_dir, f1session
            )

        if self.use_joblib and len(lap_numbers) > 1:
            results = Parallel(n_jobs=self.n_jobs, backend="loky", prefer="processes")(
                delayed(process_single_lap_job)(lap_num) for lap_num in lap_numbers
            )
        else:
            results = [process_single_lap_job(lap_num) for lap_num in lap_numbers]

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
            with open(f"{driver_dir}/laptimes.json", "w") as json_file:
                json.dump(laptimes, json_file)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()
            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )
            lap_numbers = driver_laps["LapNumber"].tolist()

            if self.use_joblib and len(lap_numbers) > self.batch_size:
                lap_batches = [
                    lap_numbers[i : i + self.batch_size]
                    for i in range(0, len(lap_numbers), self.batch_size)
                ]

                with ThreadPoolExecutor(
                    max_workers=min(4, len(lap_batches))
                ) as executor:
                    futures = [
                        executor.submit(
                            self.process_lap_batch_with_joblib,
                            session_number,
                            driver,
                            batch,
                            driver_dir,
                            f1session,
                        )
                        for batch in lap_batches
                    ]

                    for future in as_completed(futures):
                        future.result()
            else:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(
                            self.process_lap,
                            session_number,
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

    def process_testing_session(self, session_number: int) -> None:
        """Process a single testing session, extracting all telemetry data."""
        session_name = self.session_name_map.get(
            session_number, f"Session{session_number}"
        )
        logger.info(
            f"Processing Pre-Season Testing - {session_name} {'with joblib' if self.use_joblib else 'without joblib'}"
        )

        base_dir = f"Pre-Season Testing/{session_name}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_testing_session(session_number, load_telemetry=True)

            drivers_info = self.session_drivers(session_number)
            with open(f"{base_dir}/drivers.json", "w") as json_file:
                json.dump(drivers_info, json_file)

            corner_info = self.get_circuit_info(session_number)
            if corner_info:
                with open(f"{base_dir}/corners.json", "w") as json_file:
                    json.dump(corner_info, json_file)

            drivers = self.session_drivers_list(session_number)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.process_driver, session_number, driver, base_dir, f1session
                    )
                    for driver in drivers
                ]

                for future in as_completed(futures):
                    future.result()

        except Exception as e:
            logger.error(f"Error processing testing session {session_number}: {str(e)}")

    def process_all_data(self, max_workers: int = 4) -> None:
        """Process all configured testing sessions."""
        logger.info(
            f"Starting {'joblib-optimized' if self.use_joblib else 'standard'} pre-season testing extraction for {self.year}"
        )
        logger.info(
            f"Test number: {self.test_number}, Sessions: {self.session_numbers}"
        )

        if self.use_joblib:
            logger.info(
                f"Joblib settings: n_jobs={self.n_jobs}, batch_size={self.batch_size}"
            )

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for session_number in self.session_numbers:
                futures.append(
                    executor.submit(self.process_testing_session, session_number)
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(
            f"Pre-season testing extraction completed in {elapsed_time:.2f} seconds"
        )

    def clear_joblib_cache(self):
        """Clear the joblib memory cache."""
        if hasattr(memory, "clear"):
            memory.clear()
            logger.info("Joblib cache cleared")


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
        use_joblib = True
        n_jobs = -1
        batch_size = 8

        extractor = PreSeasonTelemetryExtractor(
            year=2025,
            test_number=1,
            session_numbers=[1, 2, 3],
            use_joblib=use_joblib,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

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
