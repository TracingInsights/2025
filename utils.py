import json

import requests


# make sure that year can be from 2018 to current year
class LatestData:
    def __init__(self, year):
        self.year = year
        self.data = self.get_f1_data()
        self.events = self.get_events()

    def get_f1_data(self):
        response = requests.get(
            f"https://livetiming.formula1.com/static/{self.year}/Index.json", timeout=5
        )
        if response.status_code == 200:
            try:
                data = response.content.decode("utf-8-sig")
                return json.loads(data)
            except json.JSONDecodeError as e:
                print("Failed to parse JSON data:", e)
                return None
        else:
            print("Failed to get data. Status code:", response.status_code)
            return None

    def get_events(self):
        events = []
        for meeting in self.data["Meetings"]:
            events.append(meeting["Name"])

        return events

    def get_sessions(self, event):
        sessions = []
        for meeting in self.data["Meetings"]:
            if meeting["Name"] == event:
                for session in meeting["Sessions"]:
                    sessions.append(session["Name"])

        return sessions


def team_colors(year: int) -> dict:
    team_colors = {}

    if year == 2023:
        team_colors = {
            "Red Bull Racing": "#ffe119",
            "Ferrari": "#e6194b",
            "Aston Martin": "#3cb44b",
            "Mercedes": "#00c0bf",
            "Alpine": "#f032e6",
            "Haas F1 Team": "#ffffff",
            "McLaren": "#f58231",
            "Alfa Romeo": "#800000",
            "AlphaTauri": "#dcbeff",
            "Williams": "#4363d8",
            "Red Bull Racing Honda RBPT": "#ffe119",
            "Ferrari": "#e6194b",
            "Aston Martin Aramco Mercedes": "#3cb44b",
            "Mercedes": "#00c0bf",
            "Alpine Renault": "#f032e6",
            "Haas Ferrari": "#ffffff",
            "McLaren Mercedes": "#f58231",
            "Alfa Romeo Ferrari": "#800000",
            "AlphaTauri Honda RBPT": "#dcbeff",
            "Williams Mercedes": "#4363d8",
            "Red Bull": "#ffe119",
            "Alpine F1 Team": "#f032e6",
        }
    if year == 2022:
        team_colors = {
            "Red Bull Racing": "#ffe119",
            "Ferrari": "#e6194b",
            "Aston Martin": "#3cb44b",
            "Mercedes": "#00c0bf",
            "Alpine": "#f032e6",
            "Haas F1 Team": "#ffffff",
            "McLaren": "#f58231",
            "Alfa Romeo": "#800000",
            "AlphaTauri": "#dcbeff",
            "Williams": "#4363d8",
            "Red Bull": "#ffe119",
            "Alpine F1 Team": "#f032e6",
        }

    if year == 2021:
        team_colors = {
            "Red Bull Racing": "#ffe119",
            "Mercedes": "#00c0bf",
            "Ferrari": "#e6194b",
            "Alpine": "#f032e6",
            "McLaren": "#f58231",
            "Alfa Romeo Racing": "#800000",
            "Aston Martin": "#3cb44b",
            "Haas F1 Team": "#ffffff",
            "AlphaTauri": "#dcbeff",
            "Williams": "#4363d8",
            "Red Bull": "#ffe119",
            "Alpine F1 Team": "#f032e6",
            "Alfa Romeo": "#800000",
        }

    if year == 2020:
        team_colors = {
            "Red Bull Racing": "#000099",
            "Renault": "#ffe119",
            "Racing Point": "#f032e6",
            "Mercedes": "#00c0bf",
            "Ferrari": "#e6194b",
            "McLaren": "#f58231",
            "Alfa Romeo Racing": "#800000",
            "Haas F1 Team": "#ffffff",
            "AlphaTauri": "#dcbeff",
            "Williams": "#4363d8",
            "Red Bull": "#000099",
            "Alfa Romeo": "#800000",
        }

    if year == 2019:
        team_colors = {
            "Red Bull Racing": "#000099",
            "Renault": "#ffe119",
            "Racing Point": "#f032e6",
            "Toro Rosso": "#dcbeff",
            "Mercedes": "#00c0bf",
            "Ferrari": "#e6194b",
            "McLaren": "#f58231",
            "Alfa Romeo Racing": "#800000",
            "Haas F1 Team": "#ffffff",
            "Williams": "#4363d8",
            "Red Bull": "#000099",
            "Alfa Romeo": "#800000",
        }

    if year == 2018:
        team_colors = {
            "Red Bull Racing": "#000099",
            "Renault": "#ffe119",
            "Toro Rosso": "#dcbeff",
            "Force India": "#f032e6",
            "Sauber": "#800000",
            "Mercedes": "#00c0bf",
            "Ferrari": "#e6194b",
            "McLaren": "#f58231",
            "Haas F1 Team": "#ffffff",
            "Williams": "#4363d8",
            "Red Bull": "#000099",
        }

    return team_colors