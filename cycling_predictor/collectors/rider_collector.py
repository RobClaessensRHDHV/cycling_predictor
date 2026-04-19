import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import cloudscraper
from procyclingstats import Rider
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from typing_extensions import get_args

from cycling_predictor.collectors import CPBaseCollector
from cycling_predictor.classes import CPRider, CPRiderSkill, CPRaceCategory, CPRaceYear
from cycling_predictor.maps import *


class CPRiderCollector(CPBaseCollector):
    """
    CyclingPredictor RiderCollector class.
    """
    def __init__(
            self,
            categories: List[CPRaceCategory],
            years: List[CPRaceYear]):
        """
        :param categories: Types of races from which data will be collected ('classics', 'gts', 'giro', 'tour', 'vuelta')
        :param years: Years of races from which data will be collected (2021, 2022, 2023, 2024, 2025)
        """

        self.categories: List[CPRaceCategory] = categories
        self.years: List[CPRaceYear] = years
        self.riders: List[CPRider] = list()

    def _add_rider(self, rider: CPRider):
        if rider not in self.riders:
            self.riders.append(rider)

    def _get_rider(self, name: str, team: str, url: str, raise_error: bool = False) -> Optional[CPRider]:
        for rider in self.riders:
            if rider.name == name:
                return rider
        else:
            try:
                rider_dict = Rider(url).parse()
            except (IndexError, AttributeError) as _:
                rider_dict = Rider(url).parse(exceptions_to_ignore=(IndexError, AttributeError))
                if rider_info := CPRiderInfoMap.get(name):
                    for attr, val in rider_info.items():
                        rider_dict[attr] = val
                else:
                    print(f"Rider {name} not found in RiderInfoMap.")
            except Exception as e:
                print(f"Error during retrieving rider {name}: {e}")
                if raise_error:
                    raise
                else:
                    rider_dict = {}

            rider = CPRider(
                name=name,
                team=team,
                birthdate=datetime.strptime(rider_dict.get('birthdate'), '%Y-%m-%d').date() if rider_dict.get('birthdate') else None,
                height=rider_dict.get('height', 0.0),
                weight=rider_dict.get('weight', 0.0),
            )

            self._add_rider(rider)
            return rider

    def get_riders(self):
        for race_type in self.categories:
            for year in self.years:
                for race_name in CPRaceCategoryMap[race_type]:
                    for rider_dict in self.get_startlist(race_name, year):
                        rider_name = rider_dict['rider_url'].split('/')[-1]
                        team_name = rider_dict['team_url'].split('/')[-1][:-5]
                        self._get_rider(rider_name, team_name, rider_dict['rider_url'])

    def get_co_rider_data(self):

        # Initialize the WebDriver for Edge
        driver = webdriver.Edge()

        # Open the webpage
        driver.get("https://www.cyclingoracle.com/nl/renners")

        # Wait for the page to load
        time.sleep(1)

        # Maximize the window
        driver.maximize_window()

        # Iterate over riders
        for rider in self.riders:

            # Find the search input in the 'riders_filter' div and enter the rider's name
            search_input = driver.find_element(By.CSS_SELECTOR, "#riders_filter input[type='search']")
            search_input.clear()

            # For double surnames, only search for the first surname
            if len(rider.name.split("-")) > 2 and "-van-" not in rider.name and "-de-" not in rider.name:
                search_input.send_keys(" ".join(rider.name.split("-")[:2]))
            else:
                search_input.send_keys(rider.name.replace("-", " "))
            search_input.send_keys(Keys.RETURN)

            # Wait for the search results to load
            time.sleep(2)

            # Find the specific 'odd' row for the rider
            rider_row = driver.find_element(By.CSS_SELECTOR, "tr.odd")

            try:

                # Retrieve the numeric values in cells 6 to 14 for this specific 'odd' row
                skills = dict()
                for i, skill in enumerate(get_args(CPRiderSkill)):
                    cell = rider_row.find_elements(By.TAG_NAME, "td")[i + 6]
                    skill_value = cell.text.strip()
                    skills[skill] = skill_value
                    setattr(rider, skill, int(skill_value))

                print(f"Retrieved data for rider {rider.name}:\n{skills}")

            except Exception as e:
                print(f"Error retrieving data for rider {rider.name}: {e}")
                continue

        driver.quit()

    def dumps(self):
        return {
            "categories": [category for category in self.categories],
            "years": [year for year in self.years],
            "riders": [rider.dumps() for rider in self.riders],
        }

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'CPRiderCollector':
        collector = cls(
            categories=[category for category in data.get("categories", [])],
            years=[year for year in data.get("years", [])],
        )
        for rider in data.get("riders", []):
            collector._add_rider(CPRider.loads(rider))
        return collector

    def dump(self, fp: Optional[str] = None):
        data = self.dumps()
        if not fp:
            fp = f'data/rider_collector_{"_".join(self.categories)}_{"_".join(str(year) for year in self.years)}.json'

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'CPRiderCollector':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)


if __name__ == "__main__":

    # Monkey patch requests with cloudscraper to bypass Cloudflare protections
    scraper = cloudscraper.create_scraper()
    requests.get = scraper.get

    _collector = CPRiderCollector(
        categories=['gts'],
        years=[2023, 2024, 2025],
    )

    _collector.get_riders()
    _collector.get_co_rider_data()
    for _rider in _collector.riders:
        print(_rider)

    _collector.dump()
