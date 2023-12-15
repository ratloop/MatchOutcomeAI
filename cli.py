from os import system
from time import sleep
from sys import platform, stdout
from colorama import Fore, Style

def clear():
    if platform == "linux" or platform == "linux2":
        system('clear')
    elif platform == "darwin":
        system('clear')
    elif platform == "win32":
        system('cls')

def maintitle():
    stdout.write(
        f"\x1b]2;Jag's Football Predictor\x07"
    )

def selection():

    clear()
    maintitle()

    print(Fore.YELLOW + Style.BRIGHT + "\n[0] Exit" + Style.RESET_ALL)
    print(Fore.WHITE + "\nMODULES\n" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "[1] Scrape All Datasets" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "[2] Scrape Current Season Dataset" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "[3] Make a Prediction" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "[4] View Previous Prediction" + Style.RESET_ALL)
    print("")

    module = input("INPUT\n")

    return module

def log(message: str):
    print(Fore.YELLOW + f"{message}\n"+Style.RESET_ALL)

def log_input(message: str):
    x = input(Fore.YELLOW + f"{message}\n"+Style.RESET_ALL)
    return x

def log_invalid_selection():
    print(Fore.RED + "\nInvalid Selection\n"+Style.RESET_ALL)
    sleep(0.15)