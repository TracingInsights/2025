import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import fastf1
import numpy as np
import pandas as pd
import requests

import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def convert_timedelta_to_seconds(df, column_name="LapTime"):
    """Helper function to safely convert timedelta columns to seconds."""
    if column_name in df.columns and pd.api.types.is_timedelta64_dtype(df[column_name]):
        df_copy = df.copy()
        df_copy.loc[:, column_name] = df_copy[column_name].dt.total_seconds()
        return df_copy
    return df


def laps_data(
    year: int, event: Union[str, int], session: str, driver: str
) -> Dict[str, List]:
    """Get lap data for a specific driver in a session."""
    f1session = get_session(year, event, session)
    f1session.load(telemetry=False, weather=False, messages=False)
    laps = f1session.laps

    driver_laps = laps.pick_drivers(driver)
    # Remove rows where LapTime is null
    driver_laps = driver_laps[driver_laps.LapTime.notnull()].copy()
    driver_laps = convert_timedelta_to_seconds(driver_laps)
    return {
        "time": driver_laps["LapTime"].tolist(),
        "lap": driver_laps["LapNumber"].tolist(),
        "compound": driver_laps["Compound"].tolist(),
    }


def accCalc(allLapsDriverTelemetry, Nax, Nay, Naz):
    """Calculate acceleration data from telemetry."""
    vx = allLapsDriverTelemetry["Speed"] / 3.6
    time_float = allLapsDriverTelemetry["Time"] / np.timedelta64(1, "s")
    dtime = np.gradient(time_float)
    ax = np.gradient(vx) / dtime

    for i in np.arange(1, len(ax) - 1).astype(int):
        if ax[i] > 25:
            ax[i] = ax[i - 1]

    ax_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")
    x = allLapsDriverTelemetry["X"]
    y = allLapsDriverTelemetry["Y"]
    z = allLapsDriverTelemetry["Z"]

    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    theta = np.arctan2(dy, (dx + np.finfo(float).eps))
    theta[0] = theta[1]
    theta_noDiscont = np.unwrap(theta)

    dist = allLapsDriverTelemetry["Distance"]
    ds = np.gradient(dist)
    dtheta = np.gradient(theta_noDiscont)

    for i in np.arange(1, len(dtheta) - 1).astype(int):
        if abs(dtheta[i]) > 0.5:
            dtheta[i] = dtheta[i - 1]

    C = dtheta / (ds + 0.0001)  # To avoid division by 0

    ay = np.square(vx) * C
    indexProblems = np.abs(ay) > 150
    ay[indexProblems] = 0

    ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")

    # for z
    z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
    z_theta[0] = z_theta[1]
    z_theta_noDiscont = np.unwrap(z_theta)

    ds = np.gradient(dist)
    z_dtheta = np.gradient(z_theta_noDiscont)

    for i in np.arange(1, len(z_dtheta) - 1).astype(int):
        if abs(z_dtheta[i]) > 0.5:
            z_dtheta[i] = z_dtheta[i - 1]

    z_C = z_dtheta / (ds + 0.0001)  # To avoid division by 0

    az = np.square(vx) * z_C
    indexProblems = np.abs(az) > 150
    az[indexProblems] = 0

    az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")

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
    driver_laps = driver_laps[driver_laps.LapTime.notnull()].copy()

    driver_laps = convert_timedelta_to_seconds(driver_laps)

    # Get the telemetry for lap_number
    selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

    if selected_lap.empty:
        logger.warning(f"No data for {driver} lap {lap_number} in {event} {session}")
        return None

    telemetry = selected_lap.get_telemetry()
    acc_tel = accCalc(telemetry, 3, 9, 9)
    acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

    data_key = f"{year}-{event}-{session}-{driver}-{lap_number}"

    acc_tel["DRS"] = acc_tel["DRS"].apply(lambda x: 1 if x in [10, 12, 14] else 0)
    acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x == True else 0)

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
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    logger.debug(f"Data saved to {file_path}")


def process_telemetry_data():
    """Process and save telemetry data for all specified events and sessions."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Processing telemetry data for {event} - {session}")
                drivers = session_drivers_list(YEAR, event, session)

                for driver in drivers:
                    f1session = get_session(YEAR, event, session)
                    f1session.load(telemetry=False, weather=False, messages=False)
                    laps = f1session.laps
                    driver_laps = laps.pick_drivers(driver)
                    driver_laps = driver_laps[driver_laps.LapTime.notnull()].copy()
                    driver_laps = convert_timedelta_to_seconds(driver_laps)
                    driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
                    driver_lap_numbers = driver_laps["LapNumber"].tolist()

                    for lap_number in driver_lap_numbers:
                        driver_folder = f"{event}/{session}/{driver}"
                        os.makedirs(driver_folder, exist_ok=True)

                        try:
                            telemetry = telemetry_data(
                                YEAR, event, session, driver, lap_number
                            )
                            if telemetry:
                                file_path = f"{driver_folder}/{lap_number}_tel.json"
                                save_json(telemetry, file_path)
                        except Exception as e:
                            logger.warning(
                                f"Error processing telemetry for {driver} lap {lap_number}: {e}"
                            )
                            continue
            except Exception as e:
                logger.error(f"Error processing {event} - {session}: {e}")
                continue


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


def save_lap_times():
    """Save lap times for each driver in each event and session."""
    for event in events:
        for session in sessions:
            try:
                logger.info(f"Saving lap times for {event} - {session}")
                drivers = session_drivers_list(YEAR, event, session)

                for driver in drivers:
                    driver_folder = f"{event}/{session}/{driver}"
                    os.makedirs(driver_folder, exist_ok=True)

                    laptimes = laps_data(YEAR, event, session, driver)
                    file_path = f"{driver_folder}/laptimes.json"
                    save_json(laptimes, file_path)
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

        # Process all data types
        process_telemetry_data()
        save_drivers_data()
        save_lap_times()
        save_circuit_data()

        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        # Wait and retry once
        logger.info("Waiting 5 seconds before retrying...")
        time.sleep(5)
        try:
            logger.info("Retrying data collection...")
            process_telemetry_data()
            save_drivers_data()
            save_lap_times()
            save_circuit_data()
            logger.info("Retry completed successfully")
        except Exception as e2:
            logger.error(f"Error in retry: {e2}")


if __name__ == "__main__":
    # Import any missing modules needed for the circuit info API
    try:
        import requests
    except ImportError:
        logger.error(
            "Missing required module: requests. Please install with 'pip install requests'"
        )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("f1_data_collection.log"),
            logging.StreamHandler(),
        ],
    )

    # Run the main function
    main()
