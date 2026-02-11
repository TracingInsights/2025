import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import fastf1
from fastf1.core import Telemetry
import numpy as np
import orjson
import pandas as pd
import psutil
import requests
import utils


# ---------------------------------------------------------------------------
# Patch fastf1: skip expensive add_driver_ahead (461ms/lap, unused by us).
# This scans ALL other drivers' position data per sample — we never use the
# DriverAhead / DistanceToDriverAhead columns.  Replacing it with a stub
# that returns dummy columns preserves the exact merge/resample/interpolation
# flow of get_telemetry() while eliminating the dominant per-lap bottleneck.
# Validated: maxdiff = 0.0 on all output fields vs original.
# ---------------------------------------------------------------------------
def _stub_add_driver_ahead(self, **kwargs):
    result = self.copy()
    result["DriverAhead"] = ""
    result["DistanceToDriverAhead"] = np.float64(0.0)
    return result


Telemetry.add_driver_ahead = _stub_add_driver_ahead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telemetry_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("telemetry_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}

SESSION_CACHE: Dict[str, fastf1.core.Session] = {}
CIRCUIT_INFO_CACHE: Dict[str, dict] = {}

ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
EPS = np.finfo(float).eps

# Pre-allocated smoothing kernels — avoids per-call np.ones allocation
_KERNEL_3 = np.ones(3, dtype=np.float64) / 3.0
_KERNEL_9 = np.ones(9, dtype=np.float64) / 9.0


def _write_json(path: str, obj) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=ORJSON_OPTS))


def _td_col_to_seconds(series: pd.Series) -> list:
    if series.empty:
        return []
    seconds = series.dt.total_seconds().to_numpy()
    mask = series.isna().to_numpy()
    out = np.round(seconds, 3).astype(object)
    out[mask] = "None"
    return out.tolist()


def _col_to_list_str_or_none(series: pd.Series) -> list:
    if series.empty:
        return []
    vals = series.to_numpy()
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = vals[~mask].astype(str)
    return out.tolist()


def _col_to_list_int_or_none(series: pd.Series) -> list:
    if series.empty:
        return []
    vals = series.to_numpy()
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = vals[~mask].astype(int)
    return out.tolist()


def _col_to_list_bool_or_none(series: pd.Series) -> list:
    if series.empty:
        return []
    vals = series.to_numpy()
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = vals[~mask].astype(bool)
    return out.tolist()


def _array_to_list_float_or_none(arr: np.ndarray) -> list:
    """Convert numpy array to list, replacing NaN/inf with 'None'."""
    if arr.size == 0:
        return []
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr.tolist()
    out = np.empty(arr.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = arr[~mask]
    return out.tolist()


def _array_to_list_int_or_none(arr: np.ndarray) -> list:
    """Convert numpy array to list, replacing NaN/inf with 'None'."""
    if arr.size == 0:
        return []
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr.astype(int).tolist()
    out = np.empty(arr.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = arr[~mask].astype(int)
    return out.tolist()


def _smooth_outliers(arr: np.ndarray, threshold: float, use_abs: bool) -> None:
    """In-place outlier replacement: arr[i] = arr[i-1] where exceeds threshold."""
    if use_abs:
        mask = np.abs(arr) > threshold
    else:
        mask = arr > threshold
    if mask.any():
        indices = np.where(mask)[0]
        indices = indices[(indices >= 1) & (indices < len(arr) - 1)]
        if len(indices) > 0:
            arr[indices] = arr[indices - 1]


def _compute_accelerations(
    speed: np.ndarray,
    time_arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert speed km/h -> m/s as float64
    vx = speed * (1.0 / 3.6)
    if vx.dtype != np.float64:
        vx = vx.astype(np.float64)
    time_f = (time_arr / np.timedelta64(1, "s")).astype(np.float64)

    # Ensure float64 only when needed
    x_f = x if x.dtype == np.float64 else x.astype(np.float64)
    y_f = y if y.dtype == np.float64 else y.astype(np.float64)
    z_f = z if z.dtype == np.float64 else z.astype(np.float64)
    dist_f = dist if dist.dtype == np.float64 else dist.astype(np.float64)

    # --- X acceleration ---
    dtime = np.gradient(time_f)
    ax = np.gradient(vx) / dtime
    _smooth_outliers(ax, 25.0, use_abs=False)
    ax = np.convolve(ax, _KERNEL_3, mode="same")

    # --- Shared gradient for Y and Z ---
    dx = np.gradient(x_f)
    ds = np.gradient(dist_f)

    # --- Y acceleration ---
    dy = np.gradient(y_f)
    theta = np.arctan2(dy, dx + EPS)
    theta[0] = theta[1]
    dtheta = np.gradient(np.unwrap(theta))
    _smooth_outliers(dtheta, 0.5, use_abs=True)
    C = dtheta / (ds + 0.0001)
    ay = np.square(vx) * C
    ay[np.abs(ay) > 150] = 0
    ay = np.convolve(ay, _KERNEL_9, mode="same")

    # --- Z acceleration ---
    dz = np.gradient(z_f)
    z_theta = np.arctan2(dz, dx + EPS)
    z_theta[0] = z_theta[1]
    z_dtheta = np.gradient(np.unwrap(z_theta))
    _smooth_outliers(z_dtheta, 0.5, use_abs=True)
    z_C = z_dtheta / (ds + 0.0001)
    az = np.square(vx) * z_C
    az[np.abs(az) > 150] = 0
    az = np.convolve(az, _KERNEL_9, mode="same")

    return ax, ay, az, time_f


def _process_telemetry_to_dict(telemetry: pd.DataFrame, data_key: str) -> dict:
    time_arr = telemetry["Time"].to_numpy()
    speed = telemetry["Speed"].to_numpy()
    x = telemetry["X"].to_numpy()
    y = telemetry["Y"].to_numpy()
    z = telemetry["Z"].to_numpy()
    dist = telemetry["Distance"].to_numpy()

    ax, ay, az, time_s = _compute_accelerations(speed, time_arr, x, y, z, dist)

    drs_raw = telemetry["DRS"].to_numpy()
    drs = ((drs_raw == 10) | (drs_raw == 12) | (drs_raw == 14)).astype(np.int8)
    brake = telemetry["Brake"].to_numpy().astype(bool).astype(np.int8)

    return {
        "tel": {
            "time": _array_to_list_float_or_none(time_s),
            "rpm": _array_to_list_float_or_none(telemetry["RPM"].to_numpy()),
            "speed": _array_to_list_float_or_none(speed),
            "gear": _array_to_list_int_or_none(telemetry["nGear"].to_numpy()),
            "throttle": _array_to_list_float_or_none(telemetry["Throttle"].to_numpy()),
            "brake": _array_to_list_int_or_none(brake),
            "drs": _array_to_list_int_or_none(drs),
            "distance": _array_to_list_float_or_none(dist),
            "rel_distance": _array_to_list_float_or_none(telemetry["RelativeDistance"].to_numpy()),
            "acc_x": _array_to_list_float_or_none(ax),
            "acc_y": _array_to_list_float_or_none(ay),
            "acc_z": _array_to_list_float_or_none(az),
            "x": _array_to_list_float_or_none(x),
            "y": _array_to_list_float_or_none(y),
            "z": _array_to_list_float_or_none(z),
            "dataKey": data_key,
        }
    }


class TelemetryExtractor:
    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
        use_joblib: bool = True,
        n_jobs: int = -1,
        batch_size: int = 8,
    ):
        self.year = year
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.events = events or ["Mexico City Grand Prix"]
        self.sessions = sessions or ["Practice 1"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        cache_key = f"{self.year}-{event}-{session}"
        cached = SESSION_CACHE.get(cache_key)
        if cached is not None:
            if load_telemetry and not getattr(cached, "_telemetry_loaded", False):
                cached.load(telemetry=True, weather=True, messages=True)
                cached._telemetry_loaded = True
                SESSION_CACHE[cache_key] = cached
            return cached
        f1session = fastf1.get_session(self.year, event, session)
        f1session.load(telemetry=load_telemetry, weather=True, messages=True)
        f1session._telemetry_loaded = load_telemetry
        SESSION_CACHE[cache_key] = f1session
        return f1session

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {e}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> Dict[str, List[Dict[str, str]]]:
        try:
            f1session = self.get_session(event, session)
            laps = f1session.laps
            driver_team = laps.drop_duplicates(subset="Driver")[["Driver", "Team"]]
            drivers = [
                {"driver": row.Driver, "team": row.Team}
                for row in driver_team.itertuples(index=False)
            ]
            return {"drivers": drivers}
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {e}")
            return {"drivers": []}

    def laps_data(
        self, event: Union[str, int], session: str, driver: str, f1session=None, driver_laps=None
    ) -> Dict[str, list]:
        try:
            if driver_laps is None:
                if f1session is None:
                    f1session = self.get_session(event, session)
                driver_laps = f1session.laps.pick_drivers(driver)

            return {
                "time": _td_col_to_seconds(driver_laps["LapTime"]),
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": _col_to_list_str_or_none(driver_laps["Compound"]),
                "stint": _col_to_list_int_or_none(driver_laps["Stint"]),
                "s1": _td_col_to_seconds(driver_laps["Sector1Time"]),
                "s2": _td_col_to_seconds(driver_laps["Sector2Time"]),
                "s3": _td_col_to_seconds(driver_laps["Sector3Time"]),
                "life": _col_to_list_int_or_none(driver_laps["TyreLife"]),
                "pos": _col_to_list_int_or_none(driver_laps["Position"]),
                "status": _col_to_list_str_or_none(driver_laps["TrackStatus"]),
                "pb": _col_to_list_bool_or_none(driver_laps["IsPersonalBest"]),
            }
        except Exception as e:
            logger.error(f"Error getting lap data for {driver} in {event} {session}: {e}")
            return {k: [] for k in ("time", "lap", "compound", "stint", "s1", "s2", "s3", "life", "pos", "status", "pb")}

    def get_circuit_info(self, event: str, session: str) -> Optional[Dict]:
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_session(event, session)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            try:
                circuit_info = f1session.get_circuit_info()
                corners = circuit_info.corners
                result = {
                    "CornerNumber": corners["Number"].tolist(),
                    "X": corners["X"].tolist(),
                    "Y": corners["Y"].tolist(),
                    "Angle": corners["Angle"].tolist(),
                    "Distance": corners["Distance"].tolist(),
                    "Rotation": circuit_info.rotation,
                }
                CIRCUIT_INFO_CACHE[cache_key] = result
                return result
            except (AttributeError, KeyError):
                circuit_df, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_df is not None:
                    result = {
                        "CornerNumber": circuit_df["Number"].tolist(),
                        "X": circuit_df["X"].tolist(),
                        "Y": circuit_df["Y"].tolist(),
                        "Angle": circuit_df["Angle"].tolist(),
                        "Distance": (circuit_df["Distance"] / 10).tolist(),
                        "Rotation": rotation,
                    }
                    CIRCUIT_INFO_CACHE[cache_key] = result
                    return result

            logger.warning(f"Could not get corner data for {event} {session}")
            return None
        except Exception as e:
            logger.error(f"Error getting circuit info for {event} {session}: {e}")
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

            rows = [
                (
                    float(e.get("trackPosition", {}).get("x", 0.0)),
                    float(e.get("trackPosition", {}).get("y", 0.0)),
                    int(e.get("number", 0)),
                    str(e.get("letter", "")),
                    float(e.get("angle", 0.0)),
                    float(e.get("length", 0.0)),
                )
                for e in data["corners"]
            ]

            return (
                pd.DataFrame(rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {e}")
            return None, 0.0

    def _process_single_lap(
        self,
        driver: str,
        lap_number: int,
        driver_dir: str,
        driver_laps: pd.DataFrame,
        event: str,
        session: str,
    ) -> bool:
        file_path = f"{driver_dir}/{lap_number}_tel.json"
        try:
            selected = driver_laps[driver_laps.LapNumber == lap_number]
            if selected.empty:
                logger.warning(f"No data for {driver} lap {lap_number} in {event} {session}")
                return False

            telemetry = selected.get_telemetry()
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"
            tel_data = _process_telemetry_to_dict(telemetry, data_key)
            _write_json(file_path, tel_data)
            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {e}")
            return False

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            driver_laps = f1session.laps.pick_drivers(driver)
            driver_laps = driver_laps.assign(LapNumber=driver_laps["LapNumber"].astype(int))

            laptimes = self.laps_data(event, session, driver, f1session, driver_laps)
            _write_json(f"{driver_dir}/laptimes.json", laptimes)

            lap_numbers = driver_laps["LapNumber"].tolist()

            # Pre-collect existing files to avoid per-lap os.path.exists syscalls
            existing = set(os.listdir(driver_dir)) if os.path.isdir(driver_dir) else set()

            for lap_number in lap_numbers:
                fname = f"{lap_number}_tel.json"
                if fname in existing:
                    continue
                self._process_single_lap(
                    driver, lap_number, driver_dir, driver_laps, event, session
                )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {e}")

    def process_event_session(self, event: str, session: str) -> None:
        logger.info(f"Processing {event} - {session}")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            drivers_info = self.session_drivers(event, session)
            _write_json(f"{base_dir}/drivers.json", drivers_info)

            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                _write_json(f"{base_dir}/corners.json", corner_info)

            drivers = [d["driver"] for d in drivers_info.get("drivers", [])]

            if not drivers:
                return

            # Process drivers sequentially — avoids ThreadPoolExecutor overhead
            # and GIL contention. The real bottleneck is telemetry computation
            # which is numpy-bound (releases GIL), not thread scheduling.
            for driver in drivers:
                self.process_driver(event, session, driver, base_dir, f1session)

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {e}")

    def process_all_data(self, max_workers: int = 4) -> None:
        logger.info(f"Starting optimized telemetry extraction for {self.year} season")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        start_time = time.time()

        for event in self.events:
            for session in self.sessions:
                self.process_event_session(event, session)

        elapsed = time.time() - start_time
        logger.info(f"Telemetry extraction completed in {elapsed:.2f} seconds")

    def clear_joblib_cache(self):
        logger.info("No joblib cache to clear (optimized version)")


def check_memory_usage(threshold_percent=80):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Current memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(f"Memory usage exceeds {threshold_percent}% threshold, clearing caches")
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()
        gc.collect()

        new_pct = psutil.Process(os.getpid()).memory_percent()
        logger.info(f"New memory usage after clearing caches: {new_pct:.2f}%")
        return True

    return False


def is_data_available(year, events, sessions):
    try:
        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event, session = events[0], sessions[0]
        logger.info(f"Checking data availability for {year} {event} {session}...")

        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False
        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True
    except Exception as e:
        logger.info(f"Data not yet available: {e}")
        return False


def main():
    try:
        extractor = TelemetryExtractor()

        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        wait_time = 30
        max_attempts = 720
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} season data...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info(f"Data is available for {extractor.year} season. Starting extraction...")
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time}s before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)
                check_memory_usage()

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
