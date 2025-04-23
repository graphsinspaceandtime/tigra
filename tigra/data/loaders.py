from abc import ABC, abstractmethod
from typing import Union, Literal, TextIO, Generator, List
import urllib.request
import hashlib
import os
import tempfile
import zipfile
from tigra.core.dgraph import EventSequence, Event
import re
import math
import gzip
import shutil


class _BaseDataset(ABC):
    _id: str
    _name: str
    _dataset_base_path = "../data"
    _dataset_subpath: str
    _dataset_dir: str
    _remote_url: str
    _sha256: str
    _filename: str
    _source: Union[Literal["network repository"], Literal["snap"], Literal["synthetic"]]
    _resolution: Union[
        Literal["hour"],
        Literal["day"],
        Literal["week"],
        Literal["month"],
        Literal["3-months"],
        Literal["6-months"],
        Literal["year"],
    ]
    _shrink_amount: int
    _cumulative: bool

    def __init__(
        self,
        id: str,
        name: str,
        source: Union[
            Literal["network repository"], Literal["snap"], Literal["synthetic"]
        ],
        dataset_dir: str,
        remote_url: str,
        sha256: str,
        filename: str,
        resolution: Union[
            Literal["hour"],
            Literal["day"],
            Literal["week"],
            Literal["month"],
            Literal["3-months"],
            Literal["6-months"],
            Literal["year"],
        ],
        shrink_amount: int,
        cumulative: bool,
        dataset_base_path: Union[str, None] = None,
    ):
        self._id = id
        self._name = name
        self._source = source
        self._dataset_dir = dataset_dir
        self._remote_url = remote_url
        self._sha256 = sha256
        self._filename = filename
        self._resolution = resolution
        self._shrink_amount = shrink_amount
        self._cumulative = cumulative

        if dataset_base_path != None:
            self._dataset_base_path = dataset_base_path

        dataset_subpaths_map = {
            "network repository": "netrep",
            "snap": "snap",
            "synthetic": "synthetic",
        }

        self._dataset_subpath = dataset_subpaths_map[source]

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def resolution(self):
        return self._resolution

    @property
    def shrink_amount(self):
        return self._shrink_amount

    @property
    def cumulative(self):
        return self._cumulative

    @property
    def file_path(self):
        if self._source == "network repository":
            return f"{self._dataset_base_path}/{self._dataset_subpath}/{self._dataset_dir}/{self._filename}"
        elif self._source == "snap":
            return f"{self._dataset_base_path}/{self._dataset_subpath}/{self._filename}"
        elif self._source == "synthetic":
            return f"{self._dataset_base_path}/{self._dataset_subpath}/{self._filename}"
        else:
            raise Exception(
                "Unsupported source! Is it implemented in file_path property?"
            )

    def __determine_time_resolution(self, time_resolution: str) -> int:
        # time resoultion for one day
        tr = 24 * 60 * 60
        if time_resolution == "hour":
            tr = 60 * 60
        elif time_resolution == "day":
            pass
        elif time_resolution == "week":
            tr *= 7
        elif time_resolution == "month":
            tr *= 31
        elif time_resolution == "3-months":
            tr *= 31 * 3
        elif time_resolution == "6-months":
            tr *= 31 * 6
        else:
            tr *= 31 * 12

        return tr

    def __calculate_hash(self):
        sha256_hash = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def __download_and_extract_zip(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_zip_path = temp_file.name

        print("Downloading zip file...")
        urllib.request.urlretrieve(self._remote_url, temp_zip_path)

        print("Unpacking zip file...")
        os.makedirs(f"{self._dataset_base_path}/{self._dataset_subpath}", exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(
                path=f"{self._dataset_base_path}/{self._dataset_subpath}/{self._dataset_dir}"
            )
        os.remove(temp_zip_path)

    def __download_and_ungz_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        print("Downloading gz file...")
        urllib.request.urlretrieve(self._remote_url, temp_file_path)

        print("Ungz-ing file...")
        with gzip.open(temp_file_path, "rb") as f_in, open(
            self.file_path, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(temp_file_path)

    def __hash_valid(self):
        hexdigest = self.__calculate_hash()
        if hexdigest != self._sha256:
            print("Hash is invalid!")
            print(f"Defined hash: {self._sha256}")
            print(f"Calculated hash: {hexdigest}")
            return False
        return True

    def download(self, force=False):
        if self._source == "synthetic":
            print("Synthetic datasets are not downloaded from remotes!")
            print("There is nothing to do.")
            return

        if os.path.exists(self.file_path):
            if force == False:
                if self.__hash_valid():
                    print("Hash is the same, skipping download.")
                    return
                else:
                    print("Proceeding to download.")

        print(f"Downloading dataset {self.name}...")

        if self._source == "network repository":
            self.__download_and_extract_zip()
        elif self._source == "snap":
            self.__download_and_ungz_file()
        else:
            raise Exception("Download procedure for source not implemented!")

        if not self.__hash_valid():
            raise Exception("Is dataset changed? Update sha256?")

    def _read_rows(
        self, f: TextIO, separator: str, comment: Union[str, None] = None
    ) -> Generator[List[str], None, None]:
        separator_pattern = re.compile(separator)
        comment_pattern = None if comment == None else re.compile(comment)

        for line in f:
            if comment_pattern != None and comment_pattern.match(line):
                continue

            yield [col.strip() for col in separator_pattern.split(line)]

    def _shrink_sequence(self, event_sequence: EventSequence):
        resolution = self.__determine_time_resolution(self._resolution)
        timespan = event_sequence.get_timespan()
        snaps = timespan / resolution
        snaps = math.ceil(snaps)

        snaps_to_keep = snaps + self._shrink_amount
        time_to_keep = event_sequence.events[0].timestamp + snaps_to_keep * resolution

        i = len(event_sequence.events) - 1
        while i >= 0 and event_sequence.events[i].timestamp > time_to_keep:
            i -= 1

        i += 1
        event_sequence.shrink_to(i)

    @abstractmethod
    def get_event_sequence(self, shrink: bool = False) -> EventSequence:
        events = []
        # create a list of events here
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class RadoslawEmail(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="radoslaw_email",
            name="Radoslaw Email",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="radoslaw_email",
            remote_url="https://nrvis.com/download/data/misc/radoslaw_email.zip",
            sha256="cbdc821eb8949cfa36fb71b14aa3cf745e977de1fd2de0ece099c113df364969",
            filename="radoslaw_email.edges",
            resolution="month",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            comment_pattern = r"^%"
            for row in self._read_rows(f, separator_pattern, comment_pattern):
                events.append(
                    Event(int(row[0]), int(row[1]), float(row[3]), int(row[2]))
                )
        event_sequence = EventSequence(list_of_events=events)

        if shrink and self._shrink_amount != 0:
            self._shrink_sequence(event_sequence)

        return event_sequence


class FbForum(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="fb_forum",
            name="FB Forum",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="fb-forum",
            remote_url="https://nrvis.com/download/data/dynamic/fb-forum.zip",
            sha256="4aa6ba14f2440e582e1cec3357ef9a4ac8020fe72ec6b31a7ff746e291356088",
            filename="fb-forum.edges",
            resolution="month",
            shrink_amount=-1,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class FbMessages(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="fb_messages",
            name="FB Messages",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="fb-messages",
            remote_url="https://nrvis.com/download/data/dynamic/fb-messages.zip",
            sha256="e198696b077ac26659b7e9ae59a4cbe1414c7fde77e63a40bd2271c66c377f54",
            filename="fb-messages.edges",
            resolution="month",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2][:-2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class IaContactsHypertext2009(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="ia_contacts_hypertext_2009",
            name="IA Contacts Hypertext 2009",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="ia-contacts_hypertext2009",
            remote_url="https://nrvis.com/download/data/dynamic/ia-contacts_hypertext2009.zip",
            sha256="3e3c602752bce28a2caca848ad1e357b2ba9b687f3d1d83a2de5c83f2a7e3485",
            filename="ia-contacts_hypertext2009.edges",
            resolution="day",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class IaEnronEmployees(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="ia_enron_employees",
            name="IA Enron Employees",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="ia-enron-employees",
            remote_url="https://nrvis.com/download/data/dynamic/ia-enron-employees.zip",
            sha256="550a509fdbbfa31dba52a7de85bc42ade8937bb1134c83bf5a3d8391061aeb34",
            filename="ia-enron-employees.edges",
            resolution="3-months",
            shrink_amount=-2,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            comment_pattern = r"^%"
            for row in self._read_rows(f, separator_pattern, comment_pattern):
                events.append(
                    Event(int(row[0]), int(row[1]), float(row[3]), int(row[2]))
                )
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class IaHospitalWardProximityAttr(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="ia_hospital_ward_proximity_attr",
            name="IA Hospital Ward Proximity Attributes",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="ia-hospital-ward-proximity-attr",
            remote_url="https://nrvis.com/download/data/dynamic/ia-hospital-ward-proximity-attr.zip",
            sha256="b0c4ddefe2e262e274eaca9f8d6cf1fd7324cfe89866178e5d377d710e9f803a",
            filename="ia-hospital-ward-proximity-attr.edges",
            resolution="day",
            shrink_amount=-1,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            comment_pattern = r"^%"
            for row in self._read_rows(f, separator_pattern, comment_pattern):
                events.append(
                    Event(
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        event_data={"src_class": int(row[3]), "dst_class": int(row[4])},
                    )
                )
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class IaPrimarySchoolProximityAttr(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="ia_primary_school_proximity_attr",
            name="IA Primary School Proximity Attributes",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="ia-primary-school-proximity-attr",
            remote_url="https://nrvis.com/download/data/dynamic/ia-primary-school-proximity-attr.zip",
            sha256="fffc027277827ffd145960af197edec86a2bd66ce25105622f6bb06a3dddffc5",
            filename="ia-primary-school-proximity-attr.edges",
            resolution="day",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            comment_pattern = r"^%"
            for row in self._read_rows(f, separator_pattern, comment_pattern):
                events.append(
                    Event(
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        event_data={"src_class": row[3], "dst_class": row[4]},
                    )
                )
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class IaRealityCall(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="ia_reality_call",
            name="IA Reality Call",
            source="network repository",
            dataset_base_path=base_path,
            dataset_dir="ia-reality-call",
            remote_url="https://nrvis.com/download/data/dynamic/ia-reality-call.zip",
            sha256="9b4ed613c1bbed93ecef012d1a2036945597b8fc35256dbcc150a108fbf8feef",
            filename="ia-reality-call.edges",
            resolution="week",
            shrink_amount=-1,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(
                    Event(int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                )
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class CollegeMsg(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="college_msg",
            name="College Msg",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/CollegeMsg.txt.gz",
            sha256="e00ba2415373dee52c00616065bcceaa4750e78de60d1855c76470600f10740f",
            filename="CollegeMsg.txt",
            resolution="month",
            shrink_amount=-1,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class EmailEuCoreTemporal(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="email_eu_core_temporal",
            name="Email EU Core Temporal",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
            sha256="1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f",
            filename="CollegeMsg.txt",
            resolution="month",
            shrink_amount=-1,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class SxAskUbuntu(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="sx_askubuntu",
            name="StackExchange AskUbuntu",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/sx-askubuntu.txt.gz",
            sha256="1a416ccaf5a679996156dd31290dfcca3e7dde3431171afed1b1fd616fbd31e6",
            filename="sx-askubuntu.txt",
            resolution="3-months",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class SxMathOverflow(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="sx_mathoverflow",
            name="StackExchange MathOverflow",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/sx-mathoverflow.txt.gz",
            sha256="2bf52f784cdba69730f912633aa1414b381059406ded2c38bc37658ead5b0773",
            filename="sx-mathoverflow.txt",
            resolution="3-months",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class SxSuperUser(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="sx_superuser",
            name="StackExchange SuperUser",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/sx-superuser.txt.gz",
            sha256="d99a66429d0cb69b6f61be71cac3064868cd683b251140b232f377bf42112be7",
            filename="sx-superuser.txt",
            resolution="3-months",
            shrink_amount=0,
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            self._shrink_sequence(event_sequence)

        return event_sequence


class WikiTalkTemporal(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="wiki_talk_temporal",
            name="Wiki Talk Temporal",
            source="snap",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz",
            sha256="d45012e758e26bc28f6b610f1146aee0e87728a79e69cc6d696dd180f19dff66",
            filename="wiki-talk-temporal.txt",
            resolution="day",
            shrink_amount=0,  # NOTE: undefined
            cumulative=False,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r"\W+"
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            raise Exception("Shrinking is not defined for this dataset!")

        return event_sequence


class SyntheticDBAC(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="synthetic_dbac",
            name="Synthetic Dynamic Barabasi-Albert with Communities",
            source="synthetic",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="",
            sha256="",
            filename="dbac_1000_4.events",
            resolution="month",
            shrink_amount=0,  # NOTE: undefined
            cumulative=True,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            raise Exception("Shrinking is not defined for this dataset!")

        return event_sequence


class SyntheticDBA(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="synthetic_dba",
            name="Synthetic Dynamic Barabasi-Albert",
            source="synthetic",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="",
            sha256="",
            filename="dba_1000_5_10.events",
            resolution="month",
            shrink_amount=0,  # NOTE: undefined
            cumulative=True,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            raise Exception("Shrinking is not defined for this dataset!")

        return event_sequence


class SyntheticDER(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="synthetic_der",
            name="Synthetic Dynamic Erdos-Renyi",
            source="synthetic",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="",
            sha256="",
            filename="der_1000_001.events",
            resolution="month",
            shrink_amount=0,  # NOTE: undefined
            cumulative=True,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            raise Exception("Shrinking is not defined for this dataset!")

        return event_sequence


class SyntheticDPP(_BaseDataset):
    def __init__(self, base_path: Union[str, None] = None):
        super().__init__(
            id="synthetic_dpp",
            name="Synthetic Dynamic Planted Partition",
            source="synthetic",
            dataset_base_path=base_path,
            dataset_dir="",
            remote_url="",
            sha256="",
            filename="dpp_1000_4.events",
            resolution="month",
            shrink_amount=0,  # NOTE: undefined
            cumulative=True,
        )

    def get_event_sequence(self, shrink=False):
        events = []
        with open(self.file_path, "r") as f:
            separator_pattern = r","
            for row in self._read_rows(f, separator_pattern):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        event_sequence = EventSequence(list_of_events=events)

        if shrink:
            raise Exception("Shrinking is not defined for this dataset!")

        return event_sequence


Datasets = Union[
    type(RadoslawEmail),
    type(FbForum),
    type(FbMessages),
    type(IaContactsHypertext2009),
    type(IaEnronEmployees),
    type(IaHospitalWardProximityAttr),
    type(IaPrimarySchoolProximityAttr),
    type(IaRealityCall),
    type(CollegeMsg),
    type(EmailEuCoreTemporal),
    type(SxAskUbuntu),
    type(SxMathOverflow),
    type(SxSuperUser),
    type(WikiTalkTemporal),
    type(SyntheticDBAC),
    type(SyntheticDBA),
    type(SyntheticDER),
    type(SyntheticDPP),
]


def get_dataset_by_id(id: str) -> Datasets:
    datasets: List[Datasets] = [
        RadoslawEmail,
        FbForum,
        FbMessages,
        IaContactsHypertext2009,
        IaEnronEmployees,
        IaHospitalWardProximityAttr,
        IaPrimarySchoolProximityAttr,
        IaRealityCall,
        CollegeMsg,
        EmailEuCoreTemporal,
        SxAskUbuntu,
        SxMathOverflow,
        SxSuperUser,
        WikiTalkTemporal,
        SyntheticDBAC,
        SyntheticDBA,
        SyntheticDER,
        SyntheticDPP,
    ]

    for Dataset in datasets:
        if Dataset().id == id:
            return Dataset
    raise Exception("Dataset with specified id does not exist")
