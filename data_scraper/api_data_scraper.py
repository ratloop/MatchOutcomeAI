import requests
import ujson as json

class Scraper():

    def __init__(self, year):
        self.year = year
        self.headers = {
            "X-RapidAPI-Key": "", # Enter API Key here
        }
        self.league = "39"
        self.matches = []


    def scrape(self):
        scraped = self.scrape_from_season()
        if scraped:
            for i in self.season_data:
                self.data_dict = {}
                self.data_dict["fixture_id"] = i['fixture']['id']
                self.data_dict["date"] = i['fixture']['date']
                self.data_dict["stadium"] = i['fixture']['venue']['name']
                self.data_dict["game_week"] = i['league']['round']
                self.data_dict["home_team"] = i['teams']['home']['name']
                self.data_dict["away_team"] = i['teams']['away']['name']
                self.data_dict["home_goals"] = i['goals']['home']
                self.data_dict["away_goals"] = i['goals']['away']
                self.data_dict["ht_home_goals"] = i['score']['halftime']['home']
                self.data_dict["ht_away_goals"] = i['score']['halftime']['away']
                self.scrape_season_data()
                
        self.output_json_dataset()

    def scrape_from_season(self):
        try:
            response = requests.get(f"https://v3.football.api-sports.io/fixtures?league={self.league}&season={self.year}&status=FT", headers=self.headers)

            if response.status_code != 200:
                return False
            
            json_data = response.json()

            self.season_data = json_data['response']

            return True
    
        except Exception:
            return False

    def scrape_season_data(self):

        fixture = self.data_dict["fixture_id"]

        try:
            response = requests.get(f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture}", headers=self.headers)

            if response.status_code != 200:
                return False

        except Exception:
            return False
        
        json_data = response.json()

        data = json_data['response']

        home_data = data[0]
        away_data = data[1]

        home_statistics = home_data['statistics']
        away_statistics = away_data['statistics']

        for i in home_statistics:
            type = i['type']
            if type == "Shots on Goal":
                self.data_dict["home_shots_on_target"] = i['value']
            if type == "Total Shots":
                self.data_dict["home_shots"] = i['value']
            if type == "Fouls":
                self.data_dict["home_fouls"] = i['value']
            if type == "Corner Kicks":
                self.data_dict["home_corners"] = i['value']
            if type == "Offsides":
                self.data_dict["home_offsides"] = i['value']
            if type == "Ball Possession":
                self.data_dict["home_possession"] = int(i['value'].replace("%", ""))
            if type == "Yellow Cards":
                self.data_dict["home_yellow_cards"] = i['value']
            if type == "Red Cards":
                self.data_dict["home_red_cards"] = i['value']
            if type == "Goalkeeper Saves":
                self.data_dict["home_goalkeeper_saves"] = i['value']
            if type == "Total passes":
                self.data_dict["home_attempted_passes"] = i['value']
            if type == "Passes accurate":
                self.data_dict["home_successful_passes"] = i['value']

        for i in away_statistics:
            type = i['type']
            if type == "Shots on Goal":
                self.data_dict["away_shots_on_target"] = i['value']
            if type == "Total Shots":
                self.data_dict["away_shots"] = i['value']
            if type == "Fouls":
                self.data_dict["away_fouls"] = i['value']
            if type == "Corner Kicks":
                self.data_dict["away_corners"] = i['value']
            if type == "Offsides":
                self.data_dict["away_offsides"] = i['value']
            if type == "Ball Possession":
                self.data_dict["away_possession"] = int(i['value'].replace("%", ""))
            if type == "Yellow Cards":
                self.data_dict["away_yellow_cards"] = i['value']
            if type == "Red Cards":
                self.data_dict["away_red_cards"] = i['value']
            if type == "Goalkeeper Saves":
                self.data_dict["away_goalkeeper_saves"] = i['value']
            if type == "Total passes":
                self.data_dict["away_attempted_passes"] = i['value']
            if type == "Passes accurate":
                self.data_dict["away_successful_passes"] = i['value']

        self.matches.append(self.data_dict)
        self.matchdict = {
            "season": int(self.year),
            "matches": self.matches
            }

        return True

    def output_json_dataset(self):
        
        with open(f"./data/raw_datasets/epl/{self.year}_season.json", "w+") as file:
            file.write(json.dumps(self.matchdict).replace('null', '0'))
