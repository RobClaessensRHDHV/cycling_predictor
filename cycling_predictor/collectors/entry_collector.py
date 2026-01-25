import json
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests
import cloudscraper
import procyclingstats
from procyclingstats import Race, Stage

from cycling_predictor.collectors import CPBaseCollector
from cycling_predictor.classes import CPRider, CPRace, CPStage, CPEntry, CPRaceCategory, CPRaceYear
from cycling_predictor.maps import *
from cycling_predictor.collectors.rider_collector import CPRiderCollector


class CPEntryCollector(CPBaseCollector):
    """
    CyclingPredictor EntryCollectorBase class.
    """
    def __init__(
            self,
            categories: List[CPRaceCategory],
            years: List[CPRaceYear],
            riders: List[CPRider],
            max_rank: int = -1):
        """
        :param categories: Types of races from which data will be collected ('classics', 'gts', 'giro', 'tour', 'vuelta')
        :param years: Years of races from which data will be collected (2021, 2022, 2023, 2024, 2025)
        :param riders: List of riders to consider for entries
        :param max_rank: Max rank to consider for entries, e.g. 50 to consider top 50, -1 to consider all
        """

        self.categories: List[CPRaceCategory] = categories
        self.years: List[CPRaceYear] = years
        self.riders: List[CPRider] = riders
        self.max_rank: int = max_rank
        self.races: List[CPRace] = list()
        self.stages: List[CPStage] = list()
        self.entries: List[CPEntry] = list()

    @property
    def dump_fn(self) -> str:
        if self.max_rank == -1:
            return (f"{self.__class__.__name__}_{"_".join(self.categories)}_"
                    f"{"_".join(str(year) for year in self.years)}.json")
        else:
            return (f"{self.__class__.__name__}_{"_".join(self.categories)}_"
                    f"{"_".join(str(year) for year in self.years)}_{self.max_rank}.json")

    def _add_race(self, race: CPRace):
        if race not in self.races:
            self.races.append(race)

    def _get_race(self, name: str, year: int, raise_error: bool = False) -> Optional[CPRace]:

        for race in self.races:
            if race.name == name and race.year == year:
                return race

        try:
            race_dict = Race(f"race/{name}/{year}").parse()
        except requests.exceptions.SSLError:
            print(f"SSL error during retrieving race results {name} {year}")
            if raise_error:
                raise
            else:
                return
        except procyclingstats.errors.UnexpectedParsingError:
            print(f"Unexpected parsing error during retrieving race results {name} {year}")
            if raise_error:
                raise
            else:
                return
        except Exception as e:
            print(f"Unexpected error during retrieving race results {name} {year}: {e}")
            if raise_error:
                raise
            else:
                return

        # Get startlist
        startlist = self.get_startlist(name, year, flatten=True, raise_error=raise_error)

        # Create Race
        race = CPRace(
            name=name,
            year=year,
            start_date=datetime.strptime(race_dict.get('startdate'), '%Y-%m-%d').date() if race_dict.get('startdate') else None,
            end_date=datetime.strptime(race_dict.get('enddate'), '%Y-%m-%d').date() if race_dict.get('enddate') else None,
            race_profile=CPTerrainTypeMap.get(name, None),
            startlist=startlist,
        )

        self._add_race(race)
        return race

    def _add_stage(self, stage: CPStage):
        if stage not in self.stages:
            self.stages.append(stage)

    def _get_stage(self, race: CPRace, name: str, year: int, number: int = None, raise_error: bool = False) -> CPStage:

        for stage in self.stages:
            if stage.name == name and stage.year == year and stage.stage_number == number:
                return stage

        try:
            # If number, GT stage
            if number:
                stage_dict = Stage(f"race/{name}/{year}/stage-{number}/result").parse()
            # Else, classic race
            else:
                stage_dict = Stage(f"race/{name}/{year}/result").parse()
        except requests.exceptions.SSLError:
            print(f"SSL error during retrieving race results {name} {year}")
            if raise_error:
                raise
            else:
                return
        except procyclingstats.errors.UnexpectedParsingError:
            print(f"Unexpected parsing error during retrieving race results {name} {year}")
            if raise_error:
                raise
            else:
                return
        except Exception as e:
            print(f"Unexpected error during retrieving race results {name} {year}: {e}")
            if raise_error:
                raise
            else:
                return

        # TODO: Later consider dropouts to adjust race.startlist accordingly?
        # 'dropouts' dict attribute to fill gradually during race {date: [riders]}

        # Create Stage
        stage = CPStage(
            name=name,
            year=year,
            start_date=datetime.strptime(stage_dict.get('date'), '%Y-%m-%d').date() if stage_dict.get('date') else None,
            end_date=datetime.strptime(stage_dict.get('date'), '%Y-%m-%d').date() if stage_dict.get('date') else None,
            startlist=race.startlist,
            distance=stage_dict.get('distance', 0),
            vertical_meters=stage_dict.get('vertical_meters', 0),
            profile_score=stage_dict.get('profile_score', 0),
            gradient_final_km=stage_dict.get('gradient_final_km', 0.0),
            race_startlist_quality_score=stage_dict.get('race_startlist_quality_score', (0, 0))[1],
            terrain_type=CPTerrainTypeMap.get(name, None),
            stage_type=stage_dict.get('stage_type', None),
            stage_profile=int(stage_dict.get('profile_icon')[1]) if stage_dict.get('profile_icon') else None,
            stage_number=number,
        )

        self._add_stage(stage)
        return stage

    def _add_entry(self, entry: CPEntry):
        if entry not in self.entries:
            self.entries.append(entry)

    def _get_entry(self, rider: CPRider, stage: CPStage, raise_error: bool = False) -> Optional[CPEntry]:

        for entry in self.entries:
            if entry.rider == rider and entry.stage == stage:
                return entry

        if rider.name in stage.startlist:

            # Compute age
            if rider.birthdate and stage.start_date:
                age = (stage.start_date - rider.birthdate).days // 365
            else:
                age = 0

            # Retrieve results, compute form, get rank
            rider.get_results(stage.year, raise_error=raise_error)
            form = rider.get_form(stage)
            rank = rider.get_rank(stage)

            if rank and (rank <= self.max_rank or self.max_rank == -1):
                # Create Entry
                entry = CPEntry(
                    rider=rider,
                    stage=stage,
                    rank=rank,
                    rider_age=age,
                    rider_form=form,
                )

                self._add_entry(entry)
                return entry

    @abstractmethod
    def get_entries(self):
        pass

    def get_entries_per_rider(self, rider: CPRider) -> List[CPEntry]:
        return [entry for entry in self.entries if entry.rider == rider]

    def get_entries_per_stage(self, stage: CPStage) -> List[CPEntry]:
        return [entry for entry in self.entries if entry.stage == stage]

    def get_rider(self, rider_uid: str) -> Optional[CPRider]:
        for rider in self.riders:
            if rider.uid == rider_uid:
                return rider
        return None
    
    def get_race(self, race_uid: str) -> Optional[CPRace]:
        for race in self.races:
            if race.uid == race_uid:
                return race
        
        for stage in self.stages:
            if stage.uid == race_uid:
                return stage
        
        return None

    def get_stage(self, stage_uid: str) -> Optional[CPStage]:
        for stage in self.stages:
            if stage.uid == stage_uid:
                return stage
        return None

    def dumps(self):
        return {
            "cls": self.__class__.__name__,
            "categories": [category for category in self.categories],
            "years": [year for year in self.years],
            "riders": [rider.dumps() for rider in self.riders],
            "max_rank": self.max_rank,
            "races": [race.dumps() for race in self.races],
            "stages": [stage.dumps() for stage in self.stages],
            "entries": [entry.dumps() for entry in self.entries],
        }

    @classmethod
    def loads(cls, data: Dict[str, Any], sub_cls: Optional[type] = None) -> 'CPEntryCollector':
        # Retrieve subclass if not provided
        if not sub_cls:
            for sub_cls in cls.__subclasses__():
                if sub_cls.__name__ == data.get("cls"):
                    break
            else:
                sub_cls = cls

        # Create collector
        riders = [CPRider.loads(rider) for rider in data.get("riders", [])]
        collector = sub_cls(
            categories=[category for category in data.get("categories", [])],
            years=[year for year in data.get("years", [])],
            riders=riders,
            max_rank=data.get("max_rank", -1),
        )

        # Assign additional data
        for race in data.get("races", []):
            collector._add_race(CPRace.loads(race))
        for stage in data.get("stages", []):
            collector._add_stage(CPStage.loads(stage))
        for entry in data.get("entries", []):
            collector._add_entry(CPEntry.loads(entry, collector))
        return collector

    def dump(self, fp: Optional[str] = None):
        data = self.dumps()
        if not fp:
            fp = Path(__file__).parent / 'data' / self.dump_fn

        with open(fp, 'w+') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, fp: str) -> 'CPEntryCollector':
        with open(fp, 'r') as f:
            data = json.load(f)
        return cls.loads(data)


class CPClassicEntryCollector(CPEntryCollector):
    """
    CyclingPredictor ClassicEntryCollector class.
    """
    def __init__(
            self,
            categories: List[CPRaceCategory],
            years: List[CPRaceYear],
            riders: List[CPRider],
            max_rank: int = -1):
        super().__init__(categories, years, riders, max_rank)

    def get_entries(self):
        for rider in self.riders:
            for race_type in self.categories:
                for year in self.years:
                    for race_name in CPRaceCategoryMap[race_type]:
                        race = self._get_race(race_name, year)
                        stage = self._get_stage(race, race_name, year)
                        if race and stage:
                            entry = self._get_entry(rider, stage)


class CPGTEntryCollector(CPEntryCollector):
    """
    CyclingPredictor GTEntryCollector class.
    """
    def __init__(
            self,
            categories: List[CPRaceCategory],
            years: List[CPRaceYear],
            riders: List[CPRider],
            max_rank: int = -1,
            stage_number_start: int = 1,
            stage_number_end: int = 21):
        super().__init__(categories, years, riders, max_rank)

        self.stage_number_start: int = stage_number_start
        self.stage_number_end: int = stage_number_end

    def get_entries(self):
        for rider in self.riders:
            for race_type in self.categories:
                for year in self.years:
                    for race_name in CPRaceCategoryMap[race_type]:
                        race = self._get_race(race_name, year)
                        for stage_number in range(self.stage_number_start, self.stage_number_end + 1):
                            stage = self._get_stage(race, race_name, year, stage_number)
                            if race and stage:
                                entry = self._get_entry(rider, stage)

    def dumps(self):
        data = super().dumps()
        data.update({
            "stage_number_start": self.stage_number_start,
            "stage_number_end": self.stage_number_end,
        })
        return data

    @classmethod
    def loads(cls, data: Dict[str, Any], sub_cls: Optional[type] = None) -> 'CPGTEntryCollector':
        collector = super(CPGTEntryCollector, cls).loads(data, cls)
        collector.stage_number_start = data.get("stage_number_start", 1)
        collector.stage_number_end = data.get("stage_number_end", 21)
        return collector


if __name__ == "__main__":

    # Monkey patch requests with cloudscraper to bypass Cloudflare protections
    scraper = cloudscraper.create_scraper()
    requests.get = scraper.get

    # Get rider collection
    with open('data/rider_collector_classics_2024_2025.json', 'r') as fp:
        _rider_collector = CPRiderCollector.loads(json.load(fp))
    # with open('data/rider_collector_giro_tour_vuelta_2025.json', 'r') as fp:
    #     _rider_collector = CPRiderCollector.loads(json.load(fp))

    # Classic collection
    _classic_collector = CPClassicEntryCollector(
        categories=['classics'],
        # years=[2023, 2024, 2025],
        years=[2025],
        riders=_rider_collector.riders,
        max_rank=50,
    )
    _classic_collector.get_entries()

    print('Races:')
    for _race in _classic_collector.races:
        print(_race)

    print('Entries:')
    for _entry in _classic_collector.entries:
        print(_entry)

    # Dump classic collector
    _classic_collector.dump()

    # # GT collection
    # _stage_collector = CPGTEntryCollector(
    #     # categories=['tour'],
    #     categories=['giro', 'tour', 'vuelta'],
    #     # years=[2025],
    #     years=[2023, 2024, 2025],
    #     riders=_rider_collector.riders,
    #     max_rank=100,
    # )
    # _stage_collector.get_entries()
    #
    # print('Races:')
    # for _race in _stage_collector.races:
    #     print(_race)
    #
    # print('Stages:')
    # for _stage in _stage_collector.stages:
    #     print(_stage)
    #
    # print('Entries:')
    # for _entry in _stage_collector.entries:
    #     print(_entry)
    #
    # # Dump stage collector
    # _stage_collector.dump()
