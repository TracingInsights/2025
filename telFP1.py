import concurrent.futures
import gc
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import fastf1
import numpy as np
import orjson as json_lib
import pandas as pd
import requests

import utils

pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("fastf1").setLevel(logging.WARNING)
# Prevent propagation to avoid duplicate logs
logging.getLogger("fastf1").propagate = False

fastf1.Cache.enable_cache("cache")
YEAR = 2025

events = [
    # "Pre-Season Testing",
    "Australian Grand Prix",
    # 'Chinese Grand Prix',
    # 'Japanese Grand Prix',
    # 'Bahrain Grand Prix',
    # 'Saudi Arabian Grand Prix',
    # 'Miami Grand Prix',
    # "Emilia Romagna Grand Prix",
    # 'Monaco Grand Prix',
    # 'Spanish Grand Prix',
    # 'Canadian Grand Prix',
    # 'Austrian Grand Prix',
    # 'British Grand Prix',
    # 'Belgian Grand Prix',
    # 'Hungarian Grand Prix',
    # 'Dutch Grand Prix',
    # 'Italian Grand Prix',
    # 'Azerbaijan Grand Prix',
    # 'Singapore Grand Prix',
    # 'United States Grand Prix',
    # 'Mexico City Grand Prix',
    # 'São Paulo Grand Prix',
    # 'Las Vegas Grand Prix',
    # 'Qatar Grand Prix',
    # 'Abu Dhabi Grand Prix',
]
sessions = [
    "Practice 1",
]


def events_available(year: int) -> Any:
    """Get events available for a given year."""
    data = utils.LatestData(year)
    events = data.get_events()
    return events


def sessions_available(year: int, event: Union[str, int]) -> Any:
    """Get sessions available for a given year and event."""
    event = str(event)
    data = utils.LatestData(year)
    sessions = data.get_sessions(event)
    return sessions


def get_sessions(year: int, event: str) -> List[str]:
    """Get the appropriate session list based on year and event."""
    p1_p2_p3 = ["Practice 1", "Practice 2", "Practice 3"]
    p1_p2_q_r = ["Practice 1", "Practice 2", "Qualifying", "Race"]
    p2_p3_q_r = ["Practice 2", "Practice 3", "Qualifying", "Race"]
    p3_q_r = ["Practice 3", "Qualifying", "Race"]
    p1_q_r = ["Practice 1", "Qualifying", "Race"]
    normal_sessions = [
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
        "Race",
    ]

    normal_sprint = [
        "Practice 1",
        "Qualifying",
        "Practice 2",
        "Sprint Qualifying",
        "Race",
    ]
    sprint_2022 = [
        "Practice 1",
        "Qualifying",
        "Practice 2",
        "Sprint",
        "Race",
    ]

    sprint_shootout = [
        "Practice 1",
        "Qualifying",
        "Sprint Shootout",
        "Sprint",
        "Race",
    ]
    sprint_shootout_2024 = [
        "Practice 1",
        "Sprint Shootout",
        "Sprint",
        "Qualifying",
        "Race",
    ]

    if year == 2018:
        return normal_sessions
    if year == 2019:
        if event == "Japanese Grand Prix":
            return p1_p2_q_r
        return normal_sessions
    if year == 2020:
        if event == "Styrian Grand Prix":
            return p1_p2_q_r
        if event == "Eifel Grand Prix":
            return p3_q_r
        if event == "Emilia Romagna Grand Prix":
            return p1_q_r

        return normal_sessions
    if year == 2021:
        if (
            event == "British Grand Prix"
            or event == "Italian Grand Prix"
            or event == "São Paulo Grand Prix"
        ):
            return normal_sprint
        else:
            return normal_sessions

    if year == 2022:
        if event == "Pre-Season Test":
            return p1_p2_p3
        if (
            event == "Austrian Grand Prix"
            or event == "Emilia Romagna Grand Prix"
            or event == "São Paulo Grand Prix"
        ):
            return sprint_2022
        else:
            return normal_sessions

    if year == 2023:
        if event == "Pre-Season Testing":
            return p1_p2_p3
        if event == "Hungarian Grand Prix":
            return p2_p3_q_r
        if (
            event == "Austrian Grand Prix"
            or event == "Azerbaijan Grand Prix"
            or event == "Belgium Grand Prix"
            or event == "Qatar Grand Prix"
            or event == "United States Grand Prix"
            or event == "São Paulo Grand Prix"
        ):
            return sprint_shootout
        else:
            return normal_sessions
    if year == 2024:
        if event == "Pre-Season Testing":
            return p1_p2_p3
        if (
            event == "Chinese Grand Prix"
            or event == "Miami Grand Prix"
            or event == "Austrian Grand Prix"
            or event == "United States Grand Prix"
            or event == "São Paulo Grand Prix"
            or event == "Qatar Grand Prix"
        ):
            return sprint_shootout_2024

        return normal_sessions
    if year == 2025:
        if event == "Pre-Season Testing":
            return p1_p2_p3
        if (
            event == "Chinese Grand Prix"
            or event == "Miami Grand Prix"
            or event == "Belgium Grand Prix"
            or event == "United States Grand Prix"
            or event == "São Paulo Grand Prix"
            or event == "Qatar Grand Prix"
        ):
            return sprint_shootout_2024

        return normal_sessions


@lru_cache(maxsize=128)
def get_session(year: int, event: str, session: str) -> fastf1.core.Session:
    """Get F1 session with proper handling for testing sessions."""
    if event == "Pre-Season Testing":
        return fastf1.get_testing_session(year, 1, 1)
    return fastf1.get_session(year, event, session)


def session_drivers(
    year: int, event: Union[str, int], session: str
) -> Dict[str, List[Dict[str, str]]]:
    """Get drivers available for a given year, event and session with team information."""
    f1session = get_session(year, event, session)
    f1session.load(telemetry=True, weather=False, messages=False)

    laps = f1session.laps
    team_colors = utils.team_colors(year)
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


def session_drivers_list(year: int, event: Union[str, int], session: str) -> List[str]:
    """Get list of drivers available for a given year, event and session."""
    f1session = get_session(year, event, session)
    f1session.load(telemetry=True, weather=False, messages=False)
    laps = f1session.laps
    return list(laps["Driver"].unique())


def laps_data(
    year: int, event: Union[str, int], session: str, driver: str
) -> Dict[str, List]:
    """Get lap data for a specific driver in a session."""
    f1session = get_session(year, event, session)
    f1session.load(telemetry=False, weather=False, messages=False)
    laps = f1session.laps

    driver_laps = laps.pick_drivers(driver)
    # Remove rows where LapTime is null
    driver_laps = driver_laps.dropna(subset=["LapTime"]).reset_index(drop=True).copy()
    driver_laps["LapTime"] = pd.to_numeric(
        driver_laps["LapTime"].apply(
            lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
        )
    )
    return {
        "time": driver_laps["LapTime"].tolist(),
        "lap": driver_laps["LapNumber"].tolist(),
        "compound": driver_laps["Compound"].tolist(),
    }


def accCalc(allLapsDriverTelemetry, Nax, Nay, Naz):
    """Calculate acceleration data from telemetry."""
    # Convert to numpy arrays for faster operations
    vx = allLapsDriverTelemetry["Speed"].to_numpy() / 3.6
    time_float = (
        allLapsDriverTelemetry["Time"].astype("timedelta64[ns]").astype(np.int64) / 1e9
    )
    dtime = np.gradient(time_float)
    ax = np.gradient(vx) / dtime

    # Vectorized operation instead of loop
    ax_mask = ax > 25
    if np.any(ax_mask):
        # Create a mask of indices to fix
        indices = np.where(ax_mask)[0]
        # For each index, replace with previous value
        for i in indices:
            if i > 0:  # Ensure we don't go out of bounds
                ax[i] = ax[i - 1]

    # Use more efficient convolution
    ax_smooth = np.convolve(ax, np.ones(Nax) / Nax, mode="same")

    # Extract coordinates as numpy arrays
    x = allLapsDriverTelemetry["X"].to_numpy()
    y = allLapsDriverTelemetry["Y"].to_numpy()
    z = allLapsDriverTelemetry["Z"].to_numpy()

    # Calculate gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    # Calculate theta with epsilon to avoid division by zero
    eps = np.finfo(float).eps
    theta = np.arctan2(dy, dx + eps)
    theta[0] = theta[1]  # Fix first value
    theta_noDiscont = np.unwrap(theta)

    dist = allLapsDriverTelemetry["Distance"].to_numpy()
    ds = np.gradient(dist)
    dtheta = np.gradient(theta_noDiscont)

    # Vectorized operation for dtheta correction
    dtheta_mask = np.abs(dtheta) > 0.5
    if np.any(dtheta_mask):
        indices = np.where(dtheta_mask)[0]
        for i in indices:
            if i > 0:
                dtheta[i] = dtheta[i - 1]

    # Calculate curvature with small constant to avoid division by zero
    C = dtheta / (ds + 0.0001)

    # Calculate lateral acceleration
    ay = np.square(vx) * C

    # Vectorized masking for ay
    ay[np.abs(ay) > 150] = 0
    ay_smooth = np.convolve(ay, np.ones(Nay) / Nay, mode="same")

    # Similar calculations for z-axis
    z_theta = np.arctan2(dz, dx + eps)
    z_theta[0] = z_theta[1]
    z_theta_noDiscont = np.unwrap(z_theta)
    z_dtheta = np.gradient(z_theta_noDiscont)

    # Vectorized operation for z_dtheta correction
    z_dtheta_mask = np.abs(z_dtheta) > 0.5
    if np.any(z_dtheta_mask):
        indices = np.where(z_dtheta_mask)[0]
        for i in indices:
            if i > 0:
                z_dtheta[i] = z_dtheta[i - 1]

    z_C = z_dtheta / (ds + 0.0001)
    az = np.square(vx) * z_C

    # Vectorized masking for az
    az[np.abs(az) > 150] = 0
    az_smooth = np.convolve(az, np.ones(Naz) / Naz, mode="same")

    # Assign results back to DataFrame
    allLapsDriverTelemetry["Ax"] = ax_smooth
    allLapsDriverTelemetry["Ay"] = ay_smooth
    allLapsDriverTelemetry["Az"] = az_smooth

    return allLapsDriverTelemetry


def telemetry_data(year, event, session: str, driver, lap_number):
    """Get telemetry data for a specific lap."""
    f1session = get_session(year, event, session)
    f1session.load(telemetry=True, weather=False, messages=False)
    laps = f1session.laps

    driver_laps = laps.pick_drivers(driver)
    # Filter for the specific lap number first to avoid unnecessary processing
    selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

    if selected_lap.empty:
        logger.warning(f"No data for {driver} lap {lap_number} in {event} {session}")
        return None

    # Only process the selected lap
    telemetry = selected_lap.get_telemetry()
    acc_tel = accCalc(telemetry, 3, 9, 9)

    # Convert time to seconds more efficiently
    acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

    data_key = f"{year}-{event}-{session}-{driver}-{lap_number}"

    # Vectorized operations instead of apply
    acc_tel["DRS"] = (acc_tel["DRS"].isin([10, 12, 14])).astype(int)
    acc_tel["Brake"] = acc_tel["Brake"].astype(int)

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


def get_circuit_info(year: int, circuit_key: int) -> Optional[pd.DataFrame]:
    """Get circuit information from the MultiViewer API."""
    PROTO = "https"
    HOST = "api.multiviewer.app"
    HEADERS = {"User-Agent": f"FastF1/"}

    url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{year}"

    try:
        response = fastf1.req.Cache.requests_get(url, headers=HEADERS)
        if response.status_code != 200:
            logger.debug(f"[{response.status_code}] {response.content.decode()}")
            return None

        data = response.json()
    except (
        requests.exceptions.JSONDecodeError,
        requests.exceptions.RequestException,
    ) as e:
        logger.error(f"Error fetching circuit data: {e}")
        return None

    ret = list()
    for cat in ("corners", "marshalLights", "marshalSectors"):
        rows = list()
        for entry in data[cat]:
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
        ret.append(
            pd.DataFrame(
                rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
            )
        )

    return ret[0]  # Return corners data


def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file using orjson for better performance."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as json_file:
        json_file.write(json_lib.dumps(data, option=json_lib.OPT_SERIALIZE_NUMPY))
    logger.debug(f"Data saved to {file_path}")


def process_telemetry_data():
    """Process and save telemetry data for all specified events and sessions."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Processing telemetry data for {event} - {session}")
                drivers = session_drivers_list(YEAR, event, session)

                # Load session data once per session instead of per driver
                f1session = get_session(YEAR, event, session)
                f1session.load(telemetry=False, weather=False, messages=False)
                laps = f1session.laps
                # Optimize worker count based on GitHub Actions environment
                # GitHub-hosted runners typically have 2 cores
                max_workers = min(4, os.cpu_count() or 2)
                # Process drivers in parallel
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = []
                    for driver in drivers:
                        futures.append(
                            executor.submit(
                                process_driver_telemetry,
                                YEAR,
                                event,
                                session,
                                driver,
                                laps,
                            )
                        )

                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {e}")

            except Exception as e:
                logger.error(f"Error processing {event} - {session}: {e}")
                continue


def process_driver_telemetry(year, event, session, driver, laps):
    """Process telemetry data for a specific driver."""
    try:
        driver_laps = laps.pick_drivers(driver)
        driver_laps = (
            driver_laps.dropna(subset=["LapTime"]).reset_index(drop=True).copy()
        )

        # Vectorized operations
        driver_laps["LapTime"] = pd.to_numeric(
            driver_laps["LapTime"].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )
        )
        driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
        driver_lap_numbers = driver_laps["LapNumber"].tolist()

        # Create folder once per driver
        driver_folder = f"{event}/{session}/{driver}"
        os.makedirs(driver_folder, exist_ok=True)

        # Process in smaller batches to reduce memory usage
        batch_size = 5
        for i in range(0, len(driver_lap_numbers), batch_size):
            batch_laps = driver_lap_numbers[i : i + batch_size]
            for lap_number in batch_laps:
                try:
                    telemetry = telemetry_data(year, event, session, driver, lap_number)
                    if telemetry:
                        file_path = f"{driver_folder}/{lap_number}_tel.json"
                        save_json(telemetry, file_path)
                        # Force garbage collection after each lap
                        if i % batch_size == batch_size - 1:
                            gc.collect()
                except Exception as e:
                    logger.warning(
                        f"Error processing telemetry for {driver} lap {lap_number}: {e}"
                    )
                    continue
    except Exception as e:
        logger.error(f"Error processing driver {driver}: {e}")


def save_drivers_data():
    """Save driver information for each event and session."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Saving driver data for {event} - {session}")
                drivers_info = session_drivers(YEAR, event, session)
                file_path = f"{event}/{session}/drivers.json"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                save_json(drivers_info, file_path)
                logger.info(f"Driver data saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving driver data for {event} - {session}: {e}")


def process_driver_laptimes(year, event, session, driver):
    """Process and save lap times for a specific driver."""
    try:
        driver_folder = f"{event}/{session}/{driver}"
        os.makedirs(driver_folder, exist_ok=True)

        laptimes = laps_data(year, event, session, driver)
        file_path = f"{driver_folder}/laptimes.json"
        save_json(laptimes, file_path)
    except Exception as e:
        logger.error(f"Error processing lap times for driver {driver}: {e}")


def save_lap_times():
    """Save lap times for each driver in each event and session."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Saving lap times for {event} - {session}")
                drivers = session_drivers_list(YEAR, event, session)

                # Process in batches of 5 drivers
                batch_size = 5
                for i in range(0, len(drivers), batch_size):
                    batch_drivers = drivers[i : i + batch_size]

                    # Process batch in parallel
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=batch_size
                    ) as executor:
                        futures = []
                        for driver in batch_drivers:
                            futures.append(
                                executor.submit(
                                    process_driver_laptimes,
                                    YEAR,
                                    event,
                                    session,
                                    driver,
                                )
                            )

                        # Wait for all futures to complete
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(
                                    f"Error in parallel lap time processing: {e}"
                                )

            except Exception as e:
                logger.error(f"Error saving lap times for {event} - {session}: {e}")


def save_circuit_data():
    """Save circuit and corner information for each event and session."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Saving circuit data for {event} - {session}")
                # Method 1: Using FastF1's built-in circuit info
                f1session = get_session(YEAR, event, session)
                f1session.load()

                try:
                    # Try to get circuit info from FastF1
                    circuit_info = f1session.get_circuit_info().corners
                    corner_info = {
                        "CornerNumber": circuit_info["Number"].tolist(),
                        "X": circuit_info["X"].tolist(),
                        "Y": circuit_info["Y"].tolist(),
                        "Angle": circuit_info["Angle"].tolist(),
                        "Distance": circuit_info["Distance"].tolist(),
                    }
                except Exception as e:
                    logger.warning(f"Error getting circuit info from FastF1: {e}")

                    # Method 2: Using MultiViewer API as fallback
                    try:
                        circuit_key = f1session.session_info["Meeting"]["Circuit"][
                            "Key"
                        ]
                        circuit_info = get_circuit_info(
                            year=YEAR, circuit_key=circuit_key
                        )
                        if circuit_info is not None:
                            corner_info = {
                                "CornerNumber": circuit_info["Number"].tolist(),
                                "X": circuit_info["X"].tolist(),
                                "Y": circuit_info["Y"].tolist(),
                                "Angle": circuit_info["Angle"].tolist(),
                                "Distance": (circuit_info["Distance"] / 10).tolist(),
                            }
                        else:
                            logger.error(f"Failed to get circuit info for {event}")
                            continue
                    except Exception as e2:
                        logger.error(f"Error getting circuit info from API: {e2}")
                        continue

                folder_path = f"{event}/{session}"
                os.makedirs(folder_path, exist_ok=True)
                file_path = f"{folder_path}/corners.json"
                save_json(corner_info, file_path)
                logger.info(f"Circuit data saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving circuit data for {event} - {session}: {e}")


def main():
    """Main function to run all data collection processes."""
    try:
        logger.info(f"Starting data collection for year {YEAR}")

        # Clear memory before starting
        gc.collect()

        # Process all data types
        process_telemetry_data()
        gc.collect()  # Force garbage collection between major operations

        save_drivers_data()
        gc.collect()

        save_lap_times()
        gc.collect()

        save_circuit_data()

        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        # Wait and retry once
        logger.info("Waiting 5 seconds before retrying...")
        time.sleep(5)
        gc.collect()  # Clear memory before retry
        try:
            logger.info("Retrying data collection...")
            process_telemetry_data()
            gc.collect()
            save_drivers_data()
            gc.collect()
            save_lap_times()
            gc.collect()
            save_circuit_data()
            logger.info("Retry completed successfully")
        except Exception as e2:
            logger.error(f"Error in retry: {e2}")


if __name__ == "__main__":
    # Check if running in GitHub Actions
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    # Configure logging based on environment
    if is_github_actions:
        # Simpler logging for CI environment
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # Full logging for local environment
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("f1_data_collection.log"),
                logging.StreamHandler(),
            ],
        )

    # Configure cache location based on environment
    if is_github_actions:
        # Use a directory that persists in GitHub Actions cache
        cache_dir = os.environ.get("GITHUB_WORKSPACE", "") + "/fastf1_cache"
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)

    else:
        fastf1.Cache.enable_cache("cache")

    # Import any missing modules
    try:
        import requests
    except ImportError:
        logger.error(
            "Missing required module: requests. Please install with 'pip install requests'"
        )

    # Run the main function
    main()
