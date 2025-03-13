import json
import os

import fastf1
import numpy as np
import pandas as pd

import utils

fastf1.Cache.enable_cache("cache")
YEAR = 2025


def events_available(year: int) -> any:
    # get events available for a given year
    data = utils.LatestData(year)
    events = data.get_events()
    return events


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
    # "Day 1",
]


def sessions_available(year: int, event: str | int) -> any:
    # get sessions available for a given year and event
    event = str(event)
    data = utils.LatestData(year)
    sessions = data.get_sessions(event)
    return sessions


def get_sessions(year, event):
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

def session_drivers(year: int, event: str | int, session: str) -> any:
    # get drivers available for a given year, event and session
    import fastf1

    # f1session = fastf1.get_session(year, event, session)
    f1session = fastf1.get_testing_session(2025, 1, 1)

    f1session.load(telemetry=True, weather=False, messages=False)

    laps = f1session.laps
    team_colors = utils.team_colors(year)
    # add team_colors dict to laps on Team column
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


def session_drivers_list(year: int, event: str | int, session: str) -> any:
    # get drivers available for a given year, event and session
    import fastf1

    # f1session = fastf1.get_session(year, event, session)
    f1session = fastf1.get_testing_session(2025, 1, 1)
    f1session.load(telemetry=True, weather=False, messages=False)

    laps = f1session.laps

    unique_drivers = laps["Driver"].unique()

    return list(unique_drivers)


def laps_data(year: int, event: str | int, session: str, driver: str) -> any:
    # get drivers available for a given year, event, and session
    # f1session = fastf1.get_session(year, event, session)
    f1session = fastf1.get_testing_session(2025, 1, 1)
    f1session.load(telemetry=False, weather=False, messages=False)
    laps = f1session.laps

    # add team_colors dict to laps on Team column

    # for each driver in drivers, get the Team column from laps and get the color from team_colors dict
    drivers_data = []

    driver_laps = laps.pick_driver(driver)
    driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()
    # remove rows where LapTime is null
    driver_laps = driver_laps[driver_laps.LapTime.notnull()]

    drivers_data = {
        "time": driver_laps["LapTime"].tolist(),
        "lap": driver_laps["LapNumber"].tolist(),
        "compound": driver_laps["Compound"].tolist(),
    }

    return drivers_data


# # Example usage:
# result = laps_data(2018, "Bahrain", "R", "GAS")
# result


def accCalc(allLapsDriverTelemetry, Nax, Nay, Naz):
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

    dist = allLapsDriverTelemetry["Distance"]
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
    # f1session = fastf1.get_session(year, event, session)
    f1session = fastf1.get_testing_session(2025, 1, 1)
    f1session.load(telemetry=True, weather=False, messages=False)
    laps = f1session.laps

    driver_laps = laps.pick_driver(driver)
    driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()

    # get the telemetry for lap_number
    selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

    telemetry = selected_lap.get_telemetry()

    acc_tel = accCalc(telemetry, 3, 9, 9)

    acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

    laptime = selected_lap.LapTime.values[0]
    # data_key = f"{driver} - Lap {int(lap_number)} - {year} - {session} - [{laptime}]"
    data_key = f"{year}-{event}-{session}-{driver}-{lap_number}"

    acc_tel["DRS"] = acc_tel["DRS"].apply(lambda x: 1 if x in [10, 12, 14] else 0)
    acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x == True else 0)

    telemetry_data = {
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
    return telemetry_data


while True:
    try:

        # Your list of events
        events_list = events

        # Loop through each event
        for event in events_list:
            # Get sessions for the current event
            # sessions = sessions_available(YEAR, event)

            # Loop through each session and create a folder within the event folder
            for session in sessions:
                drivers = session_drivers_list(YEAR, event, session)

                for driver in drivers:
                    # f1session = fastf1.get_session(YEAR, event, session)
                    f1session = fastf1.get_testing_session(2025, 1, 1)
                    f1session.load(telemetry=False, weather=False, messages=False)
                    laps = f1session.laps
                    driver_laps = laps.pick_driver(driver)
                    driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
                    driver_lap_numbers = round(driver_laps["LapNumber"]).tolist()

                    for lap_number in driver_lap_numbers:
                        driver_folder = f"{event}/{session}/{driver}"
                        if not os.path.exists(driver_folder):
                            os.makedirs(driver_folder)

                        try:

                            telemetry = telemetry_data(
                                YEAR, event, session, driver, lap_number
                            )

                            # print(telemetry)

                            # Specify the file path where you want to save the JSON data
                            file_path = f"{driver_folder}/{lap_number}_tel.json"

                            # Save the dictionary to a JSON file
                            with open(file_path, "w") as json_file:
                                json.dump(telemetry, json_file)
                        except:
                            continue

        def session_drivers(year: int, event: str | int, session: str) -> any:
            # get drivers available for a given year, event and session
            import fastf1

            # f1session = fastf1.get_session(year, event, session)
            f1session = fastf1.get_testing_session(2025, 1, 1)
            f1session.load(telemetry=True, weather=False, messages=False)

            laps = f1session.laps
            team_colors = utils.team_colors(year)
            # add team_colors dict to laps on Team column
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

        import json
        import os

        import utils

        # Loop through each event
        for event in events_list:

            # sessions = sessions_available(YEAR, event)

            # sessions = ['Practice 1','Practice 2','Practice 3']

            # Loop through each session and create a folder within the event folder
            for session in sessions:
                drivers = session_drivers(YEAR, event, session)

                import json

                # Specify the file path where you want to save the JSON data
                file_path = f"{event}/{session}/drivers.json"

                # Save the dictionary to a JSON file
                with open(file_path, "w") as json_file:
                    json.dump(drivers, json_file)

                print(f"Dictionary saved to {file_path}")

        def session_drivers_list(year: int, event: str | int, session: str) -> any:
            # get drivers available for a given year, event and session
            import fastf1

            # f1session = fastf1.get_session(year, event, session)
            f1session = fastf1.get_testing_session(2025, 1, 1)
            f1session.load(telemetry=True, weather=False, messages=False)

            laps = f1session.laps

            unique_drivers = laps["Driver"].unique()

            return list(unique_drivers)

        def laps_data(year: int, event: str | int, session: str, driver: str) -> any:
            # get drivers available for a given year, event, and session
            # f1session = fastf1.get_session(year, event, session)
            f1session = fastf1.get_testing_session(2025, 1, 1)
            f1session.load(telemetry=False, weather=False, messages=False)
            laps = f1session.laps

            # add team_colors dict to laps on Team column

            # for each driver in drivers, get the Team column from laps and get the color from team_colors dict
            drivers_data = []

            driver_laps = laps.pick_driver(driver)
            driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()
            # remove rows where LapTime is null
            driver_laps = driver_laps[driver_laps.LapTime.notnull()]

            drivers_data = {
                "time": driver_laps["LapTime"].tolist(),
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": driver_laps["Compound"].tolist(),
            }

            return drivers_data

        # Loop through each event
        for event in events_list:

            # # Get sessions for the current event
            # if event == "Qatar Grand Prix":
            #     sessions = ['Practice 1', 'Qualifying', 'Sprint Shootout', 'Sprint', 'Race']
            # else:
            #     sessions = sessions_available(YEAR, event)

            # Loop through each session and create a folder within the event folder
            for session in sessions:
                drivers = session_drivers_list(YEAR, event, session)

                for driver in drivers:
                    # Create a folder for the driver if it doesn't exist
                    driver_folder = f"{event}/{session}/{driver}"
                    if not os.path.exists(driver_folder):
                        os.makedirs(driver_folder)

                    laptimes = laps_data(YEAR, event, session, driver)

                    # Specify the file path where you want to save the JSON data
                    file_path = f"{driver_folder}/laptimes.json"

                    # Save the dictionary to a JSON file
                    with open(file_path, "w") as json_file:
                        json.dump(laptimes, json_file)

                    # print(f"Dictionary saved to {file_path}")

        # corners

        import json
        import os

        import fastf1

        import utils

        def sessions_available(year: int, event: str | int) -> any:
            # get sessions available for a given year and event
            event = str(event)
            data = utils.LatestData(year)
            sessions = data.get_sessions(event)
            return sessions

        for event in events:

            for session in sessions:
                # f1session = fastf1.get_session(YEAR, event, session)
                f1session = fastf1.get_testing_session(2025, 1, 1)
                f1session.load()
                circuit_info = f1session.get_circuit_info().corners
                corner_info = {
                    "CornerNumber": circuit_info["Number"].tolist(),
                    "X": circuit_info["X"].tolist(),
                    "Y": circuit_info["Y"].tolist(),
                    "Angle": circuit_info["Angle"].tolist(),
                    "Distance": circuit_info["Distance"].tolist(),
                }

                driver_folder = f"{event}/{session}"
                file_path = f"{event}/{session}/corners.json"
                if not os.path.exists(driver_folder):
                    os.makedirs(driver_folder)
                # Save the dictionary to a JSON file
                with open(file_path, "w") as json_file:
                    json.dump(corner_info, json_file)
        break
    except:
        import time

        time.sleep(5)
        continue


# corner data


from fastf1.req import Cache

PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": f"FastF1/"}


def _make_url(path: str):
    return f"{PROTO}://{HOST}{path}"


def get_circuit(*, year: int, circuit_key: int):
    """:meta private:
    Request circuit data from the MultiViewer API and return the JSON
    response."""
    url = _make_url(f"/api/v1/circuits/{circuit_key}/{year}")
    response = Cache.requests_get(url, headers=HEADERS)
    if response.status_code != 200:
        _logger.debug(f"[{response.status_code}] {response.content.decode()}")
        return None

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return None


def get_circuit_info(*, year: int, circuit_key: int):
    """:meta private:
    Load circuit information from the MultiViewer API and convert it into
    as :class:``SessionInfo`` object.

    Args:
        year: The championship year
        circuit_key: The unique circuit key (defined by the F1 livetiming API)
    """
    data = get_circuit(year=year, circuit_key=circuit_key)

    if not data:
        _logger.warning("Failed to load circuit info")
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

    rotation = float(data.get("rotation", 0.0))

    circuit_info = ret[0]

    return circuit_info


for event in events:
    for session in sessions:
        # f1session = fastf1.get_session(YEAR, event, session)
        f1session = fastf1.get_testing_session(2025, 1, 1)
        f1session.load()
        circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]
        circuit_info = get_circuit_info(year=YEAR, circuit_key=circuit_key)
        corner_info = {
            "CornerNumber": circuit_info["Number"].tolist(),
            "X": circuit_info["X"].tolist(),
            "Y": circuit_info["Y"].tolist(),
            "Angle": circuit_info["Angle"].tolist(),
            "Distance": (circuit_info["Distance"] / 10).tolist(),
        }

        driver_folder = f"{event}/{session}"
        file_path = f"{event}/{session}/corners.json"
        if not os.path.exists(driver_folder):
            os.makedirs(driver_folder)
        # Save the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(corner_info, json_file)
