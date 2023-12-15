import os
import pandas as pd
from string import capwords
from predictor import get_teams_from_season, predictor, output_previous_prediction
from cli import clear, selection, log, log_invalid_selection
from data_scraper.api_data_scraper import Scraper
import data_scraper.json_data_processor as json_data_processor

os.makedirs(f"./data/csv_datasets/epl", exist_ok=True)
os.makedirs(f"./data/raw_datasets/epl", exist_ok=True)

def scrape_data(seasons: list):
    
    log("Scraping data from the API...")

    for season in seasons:
        scraper = Scraper(season)
        scraper.scrape()

    log("Data has been scraped for the specified seasons")

    output_data()
    merge_data()

    return

def output_data():

    log("Processing Data...")

    files = os.listdir("./data/raw_datasets/epl")

    for file in files:

        file_name = os.path.join(file).split(".")[0]

        data = json_data_processor.load_raw_dataset(file_name)
        json_data_processor.output_dataset(file_name, data)

    return

def merge_data():

    log("Data has been processed and output")

    files = os.listdir("./data/csv_datasets/epl")

    df_concat = pd.concat([pd.read_csv(f"./data/csv_datasets/epl/{file}") for file in files if file != "all_seasons.csv"], ignore_index=True)
    df_concat.to_csv(f"./data/csv_datasets/epl/all_seasons.csv", index=False)

    return

def pause():
    os.system('pause')
    clear()

def interface():
    while True:
        module = selection()
        if module == "0":
            quit()
        elif module == "1":
            clear()
            scrape_data(
                [
                    "2016",
                    "2017",
                    "2018",
                    "2019",
                    "2020",
                    "2021",
                    "2022"
                    ]
                )
            pause()
        elif module == "2":
            clear()
            scrape_data(
                [
                    "2022"
                    ]
                )
            pause()
        elif module == "3":
            home_team, away_team = prediction_interface()
            predictor(home_team, away_team)
            pause()
        elif module == "4":
            output_previous_prediction()
            pause()
        else:
            log_invalid_selection()

def prediction_interface():
    teams = get_teams_from_season(season=2022)
    while True:
        clear()

        print("AVAILABLE TEAMS\n")

        log(teams)

        home_team = capwords(input("ENTER HOME TEAM\n"))
        if home_team not in teams:
            log_invalid_selection()
            continue

        away_team = capwords(input("\nENTER AWAY TEAM\n"))
        if away_team not in teams:
            log_invalid_selection()
            continue

        if home_team == away_team:
            log_invalid_selection()
            continue

        print("")

        return home_team, away_team

if __name__ == "__main__":
    interface()