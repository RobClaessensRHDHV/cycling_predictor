from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Tuple, Set
from datetime import date
from uuid import uuid4

import requests
import procyclingstats
from procyclingstats import RiderResults


CPRiderSkill = Literal['avg', 'flt', 'cob', 'hll', 'mtn', 'spr', 'itt', 'gc_', 'or_', 'ttl', 'tts', 'pr_']
CPRaceCategory = Literal['classics', 'gts', 'giro', 'tour', 'vuelta']
CPRaceYear = Literal[2021, 2022, 2023, 2024, 2025]
CPRaceProfile = Literal['sprint', 'cobbles', 'hills']
CPTStageType = Literal['RR', 'ITT', 'TTT']
CPTStageProfile = Literal[1, 2, 3, 4, 5]


class CPRider:
    """
    CyclingPredictor Rider class.
    """
    def __init__(
            self,
            name: str,
            team: str,
            category: str = 'dom',
            cost: float = 0.5,
            birthdate: date = None,
            height: float = 0.0,
            weight: float = 0.0,
            avg: int = 0,
            flt: int = 0,
            cob: int = 0,
            hll: int = 0,
            mtn: int = 0,
            spr: int = 0,
            itt: int = 0,
            gc_: int = 0,
            or_: int = 0,
            ttl: int = 0,
            tts: int = 0,
            pr_: int = 0,
            results: Dict[int, Any] = None):

        self.name = name
        self.team = team
        self.category = category
        self.cost = cost
        self.birthdate = birthdate
        self.height = height
        self.weight = weight
        self.avg = avg
        self.flt = flt
        self.cob = cob
        self.hll = hll
        self.mtn = mtn
        self.spr = spr
        self.itt = itt
        self.gc_ = gc_
        self.or_ = or_
        self.ttl = ttl
        self.tts = tts
        self.pr_ = pr_
        self.results = results
        self._uid = str(uuid4())

    def __repr__(self):
        return f"CPRider({self.name}, {self.team})"

    def __eq__(self, other):
        if isinstance(other, CPRider):
            return self.name == other.name
        return False

    @property
    def uid(self):
        return self._uid

    def get_results(self, year, raise_error: bool = False):
        if self.results and self.results.get(year):
            return

        # Get results this season
        try:
            rider_results = RiderResults(fr"rider.php?xseason={year}&racedate={year}-10-31&pracedate=smallerorequal&id={self.name}&p=results").parse()
            if self.results is None:
                self.results = dict()
            self.results[year] = rider_results['results']
        except requests.exceptions.SSLError:
            print(f"SSL error during retrieving rider results for {self.name} {year}")
            if raise_error:
                raise
        except procyclingstats.errors.UnexpectedParsingError:
            print(f"Unexpected parsing error during retrieving rider results for {self.name} {year}")
            if raise_error:
                raise
        except Exception as e:
            print(f"Unexpected error during retrieving rider results for {self.name} {year}: {e}")
            if raise_error:
                raise

    def get_form(self, race: 'CPRace', form_days: Optional[int] = 120, initial_data: bool = False):
        rider_form = 0
        for res in self.results.get(race.year, list()):

            # Possibly increment form
            if res['uci_points'] and not (initial_data and race.name in res['stage_url']):

                # Calculate number of days between result and race
                result_date = date.fromisoformat(res['date'])

                if race.start_date and result_date:
                    num_days = (race.start_date - result_date).days
                else:
                    num_days = 0
                    print(f"Number of days cannot be determined for {race}!\n"
                          f"Race date {race.start_date}, result date {result_date}")

                # If result is within amount of days, add UCI points with linear decay
                if 0 < num_days < form_days:
                    rider_form += res['uci_points'] * (1 - (num_days / form_days))

        return rider_form

    def get_rank(self, race: 'CPRace'):
        for res in self.results.get(race.year, list()):
            # NOTE: Check if distance is defined, otherwise results could be from classifications
            if res['distance'] and res['date'] == race.start_date.isoformat():
                return res['rank']

    def dumps(self):
        return {
            "name": self.name,
            "team": self.team,
            "category": self.category,
            "cost": self.cost,
            "birthdate": self.birthdate.isoformat() if self.birthdate else None,
            "height": self.height,
            "weight": self.weight,
            "avg": self.avg,
            "flt": self.flt,
            "cob": self.cob,
            "hll": self.hll,
            "mtn": self.mtn,
            "spr": self.spr,
            "itt": self.itt,
            "gc_": self.gc_,
            "or_": self.or_,
            "ttl": self.ttl,
            "tts": self.tts,
            "pr_": self.pr_,
            "uid": self.uid,
        }

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'CPRider':
        rider = cls(
            name=data.get("name", ""),
            team=data.get("team", ""),
            category=data.get("category", "dom"),
            cost=data.get("cost", 0.5),
            birthdate=date.fromisoformat(data["birthdate"]) if data.get("birthdate") else None,
            height=data.get("height", 0.0),
            weight=data.get("weight", 0.0),
            avg=data.get("avg", 0),
            flt=data.get("flt", 0),
            cob=data.get("cob", 0),
            hll=data.get("hll", 0),
            mtn=data.get("mtn", 0),
            spr=data.get("spr", 0),
            itt=data.get("itt", 0),
            gc_=data.get("gc_", 0),
            or_=data.get("or_", 0),
            ttl=data.get("ttl", 0),
            tts=data.get("tts", 0),
            pr_=data.get("pr_", 0),
            results=data.get("results", None),
        )
        rider._uid = data.get("uid", str(uuid4()))
        return rider


class CPRace(ABC):
    """
    CyclingPredictor Race class.
    """
    def __init__(
            self,
            name: str,                      # Either classic or GT stage?
            year: int = 0,
            start_date: date = None,
            end_date: date = None,
            distance: int = 0,
            vertical_meters: int = 0,
            profile_score: int = 0,
            gradient_final_km: float = 0.0,
            race_startlist_quality_score: int = 0,
            race_profile: Optional[CPRaceProfile] = None,
            startlist: List[str] = None):
        self.name = name
        self.year = year
        self.start_date = start_date
        self.end_date = end_date
        self.distance = distance
        self.vertical_meters = vertical_meters
        self.profile_score = profile_score
        self.gradient_final_km = gradient_final_km
        self.race_startlist_quality_score = race_startlist_quality_score
        self.race_profile = race_profile
        self.startlist = startlist
        self._uid = str(uuid4())

    def __repr__(self):
        if self.start_date != self.end_date:
            return f"{self.__class__.__name__}({self.name}, {self.start_date} - {self.end_date})"
        else:
            return f"{self.__class__.__name__}({self.name}, {self.start_date})"

    def __eq__(self, other):
        if isinstance(other, CPRace):
            return self.name == other.name and self.start_date == other.start_date and self.end_date == other.end_date
        return False

    @property
    def uid(self):
        return self._uid

    def dumps(self):
        return {
            "name": self.name,
            "year": self.year,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "distance": self.distance,
            "vertical_meters": self.vertical_meters,
            "profile_score": self.profile_score,
            "gradient_final_km": self.gradient_final_km,
            "race_startlist_quality_score": self.race_startlist_quality_score,
            "race_profile": self.race_profile,
            "startlist": self.startlist,
            "uid": self.uid,
        }

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'CPRace':
        race = cls(
            name=data.get("name", ""),
            year=data.get("year", 0),
            start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            distance=data.get("distance", 0),
            vertical_meters=data.get("vertical_meters", 0),
            profile_score=data.get("profile_score", 0),
            gradient_final_km=data.get("gradient_final_km", 0.0),
            race_startlist_quality_score=data.get("race_startlist_quality_score", 0),
            race_profile=data.get("race_profile", None),
            startlist=data.get("startlist", None),
        )
        race._uid = data.get("uid", str(uuid4()))
        return race

class CPStage(CPRace):
    """
    CyclingPredictor Stage class.
    """
    def __init__(
            self,
            name: str,
            year: int = 0,
            start_date: date = None,
            end_date: date = None,
            distance: int = 0,
            vertical_meters: int = 0,
            profile_score: int = 0,
            gradient_final_km: float = 0.0,
            race_startlist_quality_score: int = 0,
            race_profile: Optional[CPRaceProfile] = None,
            startlist: List[str] = None,
            stage_type: Optional[CPTStageType] = None,
            stage_profile: Optional[CPTStageProfile] = None,
            stage_number: int = 0):
        super().__init__(
            name,
            year,
            start_date,
            end_date,
            distance,
            vertical_meters,
            profile_score,
            gradient_final_km,
            race_startlist_quality_score,
            race_profile,
            startlist,
        )

        self.stage_type = stage_type
        self.stage_profile = stage_profile
        self.stage_number = stage_number

    def dumps(self):
        data = super().dumps()
        data.update({
            "stage_type": self.stage_type,
            "stage_profile": self.stage_profile,
            "stage_number": self.stage_number,
        })
        return data

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> 'CPStage':
        stage = cls(
            name=data.get("name", ""),
            year=data.get("year", 0),
            start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            distance=data.get("distance", 0),
            vertical_meters=data.get("vertical_meters", 0),
            profile_score=data.get("profile_score", 0),
            gradient_final_km=data.get("gradient_final_km", 0.0),
            race_startlist_quality_score=data.get("race_startlist_quality_score", 0),
            race_profile=data.get("race_profile", None),
            stage_type=data.get("stage_type", None),
            stage_profile=data.get("stage_profile", None),
            stage_number=data.get("stage_number", 0),
        )
        stage._uid = data.get("uid", str(uuid4()))
        return stage


class CPEntry:
    """
    CyclingPredictor Entry class.
    An entry is a combination between a rider and a race, forming the input for a prediction.
    Rider age and form are specific to the entry and are thus defined on this class.
    """
    _rider_sample_keys = (
        'avg',
        'flt',
        'cob',
        'hll',
        'mtn',
        'spr',
        'itt',
        'gc_',
        'or_',
        'ttl',
        'tts',
        'pr_',
        'height',
        'weight',
    )

    _stage_sample_keys = (
        'vertical_meters',
        'profile_score',
        'gradient_final_km',
        'race_startlist_quality_score',
    )

    _entry_sample_keys = (
        'rider_age',
        'rider_form',
    )

    def __init__(
        self,
        rider: CPRider,
        stage: CPStage,
        rank: Optional[int] = None,
        rider_age: int = 0,
        rider_form: float = 0.0,
    ):
        self.rider = rider
        self.stage = stage
        self.rank = rank
        self.rider_age = rider_age
        self.rider_form = rider_form
        self._uid = str(uuid4())

    def __repr__(self):
        if self.rank:
            return f"CPEntry({self.rider}, {self.stage}, rank={self.rank}, age={self.rider_age}, form={self.rider_form})"
        else:
            return f"CPEntry({self.rider}, {self.stage}, age={self.rider_age}, form={self.rider_form})"

    def __eq__(self, other):
        if isinstance(other, CPEntry):
            return self.rider == other.rider and self.stage == other.stage
        return False

    @property
    def uid(self):
        return self._uid

    def dumps(self):
        return {
            "rider": self.rider.uid,
            "stage": self.stage.uid,
            "rank": self.rank,
            "rider_age": self.rider_age,
            "rider_form": self.rider_form,
        }

    @classmethod
    def loads(cls, data: Dict[str, Any], collector: 'CPEntryCollector') -> 'CPEntry':
        entry = cls(
            rider=collector.get_rider(data['rider']),
            stage=collector.get_race(data['stage']),
            rank=data.get("rank", None),
            rider_age=data.get("rider_age", 0),
            rider_form=data.get("rider_form", 0.0),
        )
        entry._uid = data.get("uid", str(uuid4()))
        return entry

    def to_data(self, rider_feature_filter: Optional[Set[str]] = None, stage_feature_filter: Optional[Set[str]] = None) \
            -> Optional[Tuple[Any, ...]]:
        # Combine rider and race data in a sample tensor, and target tensor (rank)
        # Get sample tensor from rider and race data based on given keys
        sample = ([getattr(self.rider, k, 0) for k in self._rider_sample_keys if k not in (rider_feature_filter or [])] +
                  [getattr(self.stage, k, 0) for k in self._stage_sample_keys if k not in (stage_feature_filter or [])] +
                  [getattr(self, k, 0) for k in self._entry_sample_keys])

        target = self.rank or 0

        return sample, target, self.stage.uid, self.rider.uid
