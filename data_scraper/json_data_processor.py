import ujson as json
import csv
import os

columns = [
        'fixture_id',
        'season',
        'date',
        'stadium',
        'game_week',
        'home_team',
        'away_team',
        'home_goals',
        'away_goals',
        'ht_home_goals',
        'ht_away_goals',
        "home_shots_on_target",
        "home_shots",
        "home_fouls",
        "home_corners",
        "home_offsides",
        "home_possession",
        "home_yellow_cards",
        "home_red_cards",
        "home_goalkeeper_saves",
        "home_attempted_passes",
        "home_successful_passes",
        "away_shots_on_target",
        "away_shots",
        "away_fouls",
        "away_corners",
        "away_offsides",
        "away_possession",
        "away_yellow_cards",
        "away_red_cards",
        "away_goalkeeper_saves",
        "away_attempted_passes",
        "away_successful_passes",
        ]

stats = [
        'fixture_id',
        'date',
        'stadium',
        'game_week',
        'home_team',
        'away_team',
        'home_goals',
        'away_goals',
        'ht_home_goals',
        'ht_away_goals',
        "home_shots_on_target",
        "home_shots",
        "home_fouls",
        "home_corners",
        "home_offsides",
        "home_possession",
        "home_yellow_cards",
        "home_red_cards",
        "home_goalkeeper_saves",
        "home_attempted_passes",
        "home_successful_passes",
        "away_shots_on_target",
        "away_shots",
        "away_fouls",
        "away_corners",
        "away_offsides",
        "away_possession",
        "away_yellow_cards",
        "away_red_cards",
        "away_goalkeeper_saves",
        "away_attempted_passes",
        "away_successful_passes",
        ]

def load_raw_dataset(file_name):

    with open(f"./data/raw_datasets/epl/{file_name}.json", "r") as f:
        
        data = json.load(f)

    return data

def output_dataset(file_name, file_data):

    with open(f"./data/csv_datasets/epl/{file_name}.csv", "w+", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(columns)

        season = file_data["season"]

        matches = file_data["matches"]

        for match in matches:
            
            row_to_write = []

            for i in stats:
                row_to_write.append(match[i])
                
            row_to_write.insert(1, season)

            writer.writerow(row_to_write)

def output_data():

    files = os.listdir("./data/raw_datasets/epl")

    for file in files:

        file_name = os.path.join(file).split(".")[0]

        data = load_raw_dataset(file_name)
        output_dataset(file_name, data)