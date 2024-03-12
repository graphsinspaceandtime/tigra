import typing
import sys
from pathlib import Path
import re

sys.path.append("..")

from tigra.core.dgraph import EventSequence, Event

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


def get_data(dataset: Dataset) -> EventSequence:
    """
    Read a specified dataset and return an EventSequence object.
    """
    if dataset == "radoslaw-email":
        events_generator = __read_radoslaw(filename_map[dataset])
    elif dataset in [
        "CollegeMsg",
        "email-Eu-core-temporal",
        "sx-askubuntu",
        "sx-mathoverflow",
        "sx-superuser",
        "wiki-talk-temporal",
    ]:
        events_generator = __read_three_whitespace_separated_columns(
            filename_map[dataset]
        )
    else:
        raise Exception("{} dataset is not available".format(dataset))

    return EventSequence(events_generator=events_generator)


def get_keys() -> typing.List[Dataset]:
    return list(filename_map.keys())


def __read_radoslaw(filename: Path) -> typing.Generator[typing.List[Event], None, None]:
    with open(filename, "r") as f:
        separator_pat = r"\W+"
        comment_pat = r"^%"
        for row in __read_rows(f, separator_pat, comment_pat):
            yield Event(int(row[0]), int(row[1]), float(row[3]), int(row[2]))


def __read_three_whitespace_separated_columns(
    filename: Path,
) -> typing.Generator[typing.List[Event], None, None]:
    events = []
    with open(filename, "r") as f:
        separator_pat = r"\W+"
        for row in __read_rows(f, separator_pat):
            yield Event(int(row[0]), int(row[1]), int(row[2]))


def __read_rows(
    f: typing.TextIO, separator: str, comment: typing.Union[str, None] = None
) -> typing.Generator[typing.List[str], None, None]:
    if separator == None:
        raise Exception("separator expression cannot be None")

    separator_pat = re.compile(separator)
    comment_pat = None if comment == None else re.compile(comment)

    for line in f:
        if comment_pat != None and comment_pat.match(line):
            continue

        yield [col.strip() for col in separator_pat.split(line)]
