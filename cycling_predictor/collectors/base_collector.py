from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
import procyclingstats
from procyclingstats import RaceStartlist


class CPBaseCollector(ABC):
    """
    CyclingPredictor EntryCollectorBase class.
    """

    @abstractmethod
    def dumps(self):
        pass

    @classmethod
    @abstractmethod
    def loads(cls, data: Dict[str, Any], sub_cls: Optional[type] = None) -> 'CPBaseCollector':
        pass

    @abstractmethod
    def dump(self, fp: Optional[str] = None):
        pass

    @classmethod
    @abstractmethod
    def load(cls, fp: str) -> 'CPBaseCollector':
        pass

    @staticmethod
    def get_startlist(race: str, year: int, flatten: bool = False, raise_error: bool = False) -> List[Any]:
        try:
            startlist = RaceStartlist(f"race/{race}/{year}/startlist").startlist()
            if flatten:
                return [r['rider_url'].split('/')[-1] for r in startlist]
            else:
                return startlist
        except requests.exceptions.SSLError:
            print(f"SSL error during retrieving startlist of race/{race}/{year}")
            if raise_error:
                raise
            else:
                return list()
        except procyclingstats.errors.UnexpectedParsingError:
            print(f"Unexpected parsing error during retrieving startlist of race/{race}/{year}")
            if raise_error:
                raise
            else:
                return list()
        except Exception as e:
            print(f"Unexpected error during retrieving startlist of race/{race}/{year}: {e}")
            if raise_error:
                raise
            else:
                return list()
