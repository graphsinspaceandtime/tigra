import typing
import sys
from pathlib import Path
import re
import math

sys.path.append("..")

from tigra.core.dgraph import EventSequence, Event

Dataset = typing.Literal[
    "radoslaw-email",
    "fb-forum",
    "fb-messages",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-hospital-ward-proximity-attr",
    "ia-primary-school-proximity-attr",
    "ia-reality-call",
    "CollegeMsg",
    "email-Eu-core-temporal",
    "sx-askubuntu",
    "sx-mathoverflow",
    "sx-superuser",
    "wiki-talk-temporal",
    "synthetic-dba",
    "synthetic-dbac",
    "synthetic-der",
    "synthetic-dpp",
]


class DatasetMetadata(typing.TypedDict):
    path: Path
    shrink: typing.Union[int, None]
    resolution: typing.Union[str, None]


class Databank:
    """
    Class for easier loading of datasets for experiment scripts. It is designed
    for supporting both local execution of experiments using datasets stored in
    the data directory and execution on a compute cluster where datasets are
    sotred in a shared location.

    Default dataset path is the data directory in the root of the repository.
    """

    def __init__(self, path: typing.Union[str, Path, None] = None) -> None:
        """
        Create an instance of Databank loader.

        Example use:
        ```python
        from data.bank import Databank

        databank = Databank()

        # or in case if you want to specify a custom path

        databank = Databank("/path/to/datasets")
        ```
        """
        if isinstance(path, str):
            self.__data_path = Path(path)
        elif isinstance(path, Path):
            self.__data_path = path
        else:
            self.__data_path = Path(__file__).parent

        self.__netrep_path = self.__data_path / "netrep"
        self.__snap_path = self.__data_path / "snap"
        self.__synthetic_path = self.__data_path / "synthetic"

        self.__dataset_metadata: typing.Dict[str, DatasetMetadata] = {
            "radoslaw-email": {
                "path": self.__netrep_path / "radoslaw_email" / "radoslaw_email.edges",
                "shrink": 0,
                "resolution": "month",
            },
            "fb-forum": {
                "path": self.__netrep_path / "fb-forum" / "fb-forum.edges",
                "shrink": -1,
                "resolution": "month",
            },
            "fb-messages": {
                "path": self.__netrep_path / "fb-messages" / "fb-messages.edges",
                "shrink": 0,
                "resolution": "month",
            },
            "ia-contacts_hypertext2009": {
                "path": self.__netrep_path
                / "ia-contacts_hypertext2009"
                / "ia-contacts_hypertext2009.edges",
                "shrink": 0,
                "resolution": "day",
            },
            "ia-enron-employees": {
                "path": self.__netrep_path
                / "ia-enron-employees"
                / "ia-enron-employees.edges",
                "shrink": -2,
                "resolution": "3-months",
            },
            "ia-hospital-ward-proximity-attr": {
                "path": self.__netrep_path
                / "ia-hospital-ward-proximity-attr"
                / "ia-hospital-ward-proximity-attr.edges",
                "shrink": -1,
                "resolution": "day",
            },
            "ia-primary-school-proximity-attr": {
                "path": self.__netrep_path
                / "ia-primary-school-proximity-attr"
                / "ia-primary-school-proximity-attr.edges",
                "shrink": 0,
                "resolution": "day",
            },
            "ia-reality-call": {
                "path": self.__netrep_path
                / "ia-reality-call"
                / "ia-reality-call.edges",
                "shrink": -1,
                "resolution": "week",
            },
            "CollegeMsg": {
                "path": self.__snap_path / "CollegeMsg.txt",
                "shrink": -1,
                "resolution": "month",
            },
            "email-Eu-core-temporal": {
                "path": self.__snap_path / "email-Eu-core-temporal.txt",
                "shrink": -1,
                "resolution": "month",
            },
            "sx-askubuntu": {
                "path": self.__snap_path / "sx-askubuntu.txt",
                "shrink": 0,
                "resolution": "3-months",
            },
            "sx-mathoverflow": {
                "path": self.__snap_path / "sx-mathoverflow.txt",
                "shrink": 0,
                "resolution": "3-months",
            },
            "sx-superuser": {
                "path": self.__snap_path / "sx-superuser.txt",
                "shrink": 0,
                "resolution": "3-monhts",
            },
            "wiki-talk-temporal": {
                "path": self.__snap_path / "wiki-talk-temporal.txt",
                "shrink": None,
                "resolution": None,
            },
            "synthetic-dba": {
                "path": self.__synthetic_path / "dba_1000_5_10.events",
                "shrink": None,
                "resolution": "month",
            },
            "synthetic-dbac": {
                "path": self.__synthetic_path / "dbac_1000_4.events",
                "shrink": None,
                "resolution": "month",
            },
            "synthetic-der": {
                "path": self.__synthetic_path / "der_1000_001.events",
                "shrink": None,
                "resolution": "month",
            },
            "synthetic-dpp": {
                "path": self.__synthetic_path / "dpp_1000_4.events",
                "shrink": None,
                "resolution": "month",
            },
        }

        self.__filename_map: typing.Dict[str, Path] = {
            "radoslaw-email": self.__netrep_path
            / "radoslaw_email"
            / "radoslaw_email.edges",
            "fb-forum": self.__netrep_path / "fb-forum" / "fb-forum.edges",
            "fb-messages": self.__netrep_path / "fb-messages" / "fb-messages.edges",
            "ia-contacts_hypertext2009": self.__netrep_path
            / "ia-contacts_hypertext2009"
            / "ia-contacts_hypertext2009.edges",
            "ia-enron-employees": self.__netrep_path
            / "ia-enron-employees"
            / "ia-enron-employees.edges",
            "ia-hospital-ward-proximity-attr": self.__netrep_path
            / "ia-hospital-ward-proximity-attr"
            / "ia-hospital-ward-proximity-attr.edges",
            "ia-primary-school-proximity-attr": self.__netrep_path
            / "ia-primary-school-proximity-attr"
            / "ia-primary-school-proximity-attr.edges",
            "ia-reality-call": self.__netrep_path
            / "ia-reality-call"
            / "ia-reality-call.edges",
            "CollegeMsg": self.__snap_path / "CollegeMsg.txt",
            "email-Eu-core-temporal": self.__snap_path / "email-Eu-core-temporal.txt",
            "sx-askubuntu": self.__snap_path / "sx-askubuntu.txt",
            "sx-mathoverflow": self.__snap_path / "sx-mathoverflow.txt",
            "sx-superuser": self.__snap_path / "sx-superuser.txt",
            "wiki-talk-temporal": self.__snap_path / "wiki-talk-temporal.txt",
            "synthetic-dba": self.__synthetic_path / "dba_1000_5_10.events",
            "synthetic-dbac": self.__synthetic_path / "dbac_1000_4.events",
            "synthetic-der": self.__synthetic_path / "der_1000_001.events",
            "synthetic-dpp": self.__synthetic_path / "dpp_1000_4.events",
        }

    def get_data(self, dataset: Dataset, shrink_sequence=False) -> EventSequence:
        """
        Read a specified dataset and return an EventSequence object.

        Example use:
        ```python
        es = databank.get_data("radoslaw-email")
        ```
        """
        if dataset in ["radoslaw-email", "ia-enron-employees"]:
            es = self.__read_wsp_vert_weig_time(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset in ["fb-forum", "ia-contacts_hypertext2009"]:
            es = self.__read_csv_vert_time(self.__dataset_metadata[dataset]["path"])
        elif dataset == "fb-messages":
            es = self.__read_csv_vert_float_time(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset == "ia-hospital-ward-proximity-attr":
            es = self.__read_csv_vert_time_attr(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset == "ia-primary-school-proximity-attr":
            # TODO: Za ovaj mozda uraditi diskretizaciju atributa?
            es = self.__read_csv_vert_time_str_attr(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset == "ia-reality-call":
            es = self.__read_csv_vert_time_weig(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset in [
            "CollegeMsg",
            "email-Eu-core-temporal",
            "sx-askubuntu",
            "sx-mathoverflow",
            "sx-superuser",
            "wiki-talk-temporal",
        ]:
            es = self.__read_three_whitespace_separated_columns(
                self.__dataset_metadata[dataset]["path"]
            )
        elif dataset in [
            "synthetic-dba",
            "synthetic-dbac",
            "synthetic-der",
            "synthetic-dpp",
        ]:
            es = self.__read_csv_vert_time(self.__filename_map[dataset])
        else:
            raise Exception("{} dataset is not available".format(dataset))

        if not shrink_sequence:
            return es

        if self.__dataset_metadata[dataset]["resolution"] == None:
            raise Exception(
                f"{dataset} dataset doesn't have the timeresolution defined"
            )

        tr = self.__determine_time_resolution(
            self.__dataset_metadata[dataset]["resolution"]  # type: ignore
        )

        timespan = es.get_timespan()
        snaps = timespan / tr
        snaps = math.ceil(snaps)

        snaps_to_keep = snaps + self.__dataset_metadata[dataset]["shrink"]
        time_to_keep = es.events[0].timestamp + snaps_to_keep * tr

        i = len(es.events) - 1
        while i >= 0 and es.events[i].timestamp > time_to_keep:
            i -= 1

        i += 1
        es.shrink_to(i)

        return es

    def get_keys(self) -> typing.List[Dataset]:
        """
        Get a list of dataset keys. The paths are stored in a dictionary and
        by specifying the dictionary key, you pick which dataset is used.

        Example use:
        ```python
        for key in databank.get_keys():
            es = databank.get_data(key)
        ```
        """
        return typing.cast(typing.List[Dataset], list(self.__filename_map.keys()))

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

    def __read_wsp_vert_weig_time(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r"\W+"
            comment_pat = r"^%"
            for row in self.__read_rows(f, separator_pat, comment_pat):
                events.append(
                    Event(int(row[0]), int(row[1]), float(row[3]), int(row[2]))
                )
        return EventSequence(list_of_events=events)

    def __read_three_whitespace_separated_columns(
        self, filename: Path
    ) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r"\W+"
            for row in self.__read_rows(f, separator_pat):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        return EventSequence(list_of_events=events)

    def __read_csv_vert_time(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r","
            for row in self.__read_rows(f, separator_pat):
                events.append(Event(int(row[0]), int(row[1]), int(row[2])))
        return EventSequence(list_of_events=events)

    def __read_csv_vert_float_time(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r","
            for row in self.__read_rows(f, separator_pat):
                events.append(Event(int(row[0]), int(row[1]), int(row[2][:-2])))
        return EventSequence(list_of_events=events)

    def __read_csv_vert_time_attr(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r","
            comment_pat = r"^%"
            for row in self.__read_rows(f, separator_pat, comment_pat):
                events.append(
                    Event(
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        event_data={"src_class": int(row[3]), "dst_class": int(row[4])},
                    )
                )
        return EventSequence(list_of_events=events)

    def __read_csv_vert_time_str_attr(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r","
            comment_pat = r"^%"
            for row in self.__read_rows(f, separator_pat, comment_pat):
                events.append(
                    Event(
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        event_data={"src_class": row[3], "dst_class": row[4]},
                    )
                )
        return EventSequence(list_of_events=events)

    def __read_csv_vert_time_weig(self, filename: Path) -> EventSequence:
        events = []
        with open(filename, "r") as f:
            separator_pat = r"\W+"
            for row in self.__read_rows(f, separator_pat):
                events.append(
                    Event(int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                )
        return EventSequence(list_of_events=events)

    def __read_rows(
        self, f: typing.TextIO, separator: str, comment: typing.Union[str, None] = None
    ) -> typing.Generator[typing.List[str], None, None]:
        if separator == None:
            raise Exception("separator expression cannot be None")

        separator_pat = re.compile(separator)
        comment_pat = None if comment == None else re.compile(comment)

        for line in f:
            if comment_pat != None and comment_pat.match(line):
                continue

            yield [col.strip() for col in separator_pat.split(line)]
