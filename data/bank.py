import typing
import sys
from pathlib import Path
import re

sys.path.append("..")

from tigra.core.dgraph import EventSequence, Event


class Databank:
    """
    Class for easier loading of datasets for experiment scripts. It is designed
    for supporting both local execution of experiments using datasets stored in
    the data directory and execution on a compute cluster where datasets are
    sotred in a shared location.

    Default dataset path is the data directory in the root of the repository.
    """

    __Dataset = typing.Literal[
        "radoslaw-email",
        "CollegeMsg",
        "email-Eu-core-temporal",
        "sx-askubuntu",
        "sx-mathoverflow",
        "sx-superuser",
        "wiki-talk-temporal",
    ]

    __data_path = Path(__file__).parent
    __netrep_path = __data_path / "netrep"
    __snap_path = __data_path / "snap"

    __filename_map: typing.Dict[__Dataset, Path] = {
        "radoslaw-email": __netrep_path / "radoslaw_email" / "radoslaw_email.edges",
        "CollegeMsg": __snap_path / "CollegeMsg.txt",
        "email-Eu-core-temporal": __snap_path / "email-Eu-core-temporal.txt",
        "sx-askubuntu": __snap_path / "sx-askubuntu.txt",
        "sx-mathoverflow": __snap_path / "sx-mathoverflow.txt",
        "sx-superuser": __snap_path / "sx-superuser.txt",
        "wiki-talk-temporal": __snap_path / "wiki-talk-temporal.txt",
    }

    def __init__(self, path: typing.Union[str, Path, None]) -> None:
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

        self.__filename_map: typing.Dict[self.__Dataset, Path] = {
            "radoslaw-email": self.__netrep_path
            / "radoslaw_email"
            / "radoslaw_email.edges",
            "CollegeMsg": self.__snap_path / "CollegeMsg.txt",
            "email-Eu-core-temporal": self.__snap_path / "email-Eu-core-temporal.txt",
            "sx-askubuntu": self.__snap_path / "sx-askubuntu.txt",
            "sx-mathoverflow": self.__snap_path / "sx-mathoverflow.txt",
            "sx-superuser": self.__snap_path / "sx-superuser.txt",
            "wiki-talk-temporal": self.__snap_path / "wiki-talk-temporal.txt",
        }

    def get_data(self, dataset: __Dataset) -> EventSequence:
        """
        Read a specified dataset and return an EventSequence object.

        Example use:
        ```python
        es = databank.get_data("radoslaw-email")
        ```
        """
        if dataset == "radoslaw-email":
            return self.__read_radoslaw(self.__filename_map[dataset])
        if dataset in [
            "CollegeMsg",
            "email-Eu-core-temporal",
            "sx-askubuntu",
            "sx-mathoverflow",
            "sx-superuser",
            "wiki-talk-temporal",
        ]:
            return self.__read_three_whitespace_separated_columns(
                self.__filename_map[dataset]
            )
        else:
            raise Exception("{} dataset is not available".format(dataset))

    def get_keys(self) -> typing.List[__Dataset]:
        """
        Get a list of dataset keys. The paths are stored in a dictionary and
        by specifying the dictionary key, you pick which dataset is used.

        Example use:
        ```python
        for key in databank.get_keys():
            es = databank.get_data(key)
        ```
        """
        return list(self.__filename_map.keys())

    def __read_radoslaw(self, filename: Path) -> EventSequence:
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
