import typing
import sys
from pathlib import Path
import re

sys.path.append("..")

from tigra.core.dgraph import EventSequence, Event


class Databank:
    Dataset = typing.Literal[
        "radoslaw-email",
        "CollegeMsg",
        "email-Eu-core-temporal",
        "sx-askubuntu",
        "sx-mathoverflow",
        "sx-superuser",
        "wiki-talk-temporal",
    ]

    data_path = Path(__file__).parent
    netrep_path = data_path / "netrep"
    snap_path = data_path / "snap"

    filename_map: typing.Dict[Dataset, Path] = {
        "radoslaw-email": netrep_path / "radoslaw_email" / "radoslaw_email.edges",
        "CollegeMsg": snap_path / "CollegeMsg.txt",
        "email-Eu-core-temporal": snap_path / "email-Eu-core-temporal.txt",
        "sx-askubuntu": snap_path / "sx-askubuntu.txt",
        "sx-mathoverflow": snap_path / "sx-mathoverflow.txt",
        "sx-superuser": snap_path / "sx-superuser.txt",
        "wiki-talk-temporal": snap_path / "wiki-talk-temporal.txt",
    }

    def __init__(self, path: typing.Union[str, Path, None]) -> None:
        if isinstance(path, str):
            self.data_path = Path(path)
        elif isinstance(path, Path):
            self.data_path = path
        else:
            self.data_path = Path(__file__).parent

        self.netrep_path = self.data_path / "netrep"
        self.snap_path = self.data_path / "snap"

        self.filename_map: typing.Dict[self.Dataset, Path] = {
            "radoslaw-email": self.netrep_path
            / "radoslaw_email"
            / "radoslaw_email.edges",
            "CollegeMsg": self.snap_path / "CollegeMsg.txt",
            "email-Eu-core-temporal": self.snap_path / "email-Eu-core-temporal.txt",
            "sx-askubuntu": self.snap_path / "sx-askubuntu.txt",
            "sx-mathoverflow": self.snap_path / "sx-mathoverflow.txt",
            "sx-superuser": self.snap_path / "sx-superuser.txt",
            "wiki-talk-temporal": self.snap_path / "wiki-talk-temporal.txt",
        }

    def get_data(self, dataset: Dataset) -> EventSequence:
        """
        Read a specified dataset and return an EventSequence object.
        """
        if dataset == "radoslaw-email":
            return self.__read_radoslaw(self.filename_map[dataset])
        if dataset in [
            "CollegeMsg",
            "email-Eu-core-temporal",
            "sx-askubuntu",
            "sx-mathoverflow",
            "sx-superuser",
            "wiki-talk-temporal",
        ]:
            return self.__read_three_whitespace_separated_columns(
                self.filename_map[dataset]
            )
        else:
            raise Exception("{} dataset is not available".format(dataset))

    def get_keys(self) -> typing.List[Dataset]:
        return list(self.filename_map.keys())

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
