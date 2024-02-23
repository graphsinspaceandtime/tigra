import typing
import sys
from pathlib import Path
import re

sys.path.append("..")

from tigra.core.dgraph import EventSequence, Event

Datasets = typing.Literal["radoslaw-email"]

filename_map = {
    "radoslaw-email": Path(__file__).parent / "netrep" / "ia-radoslaw-email.edges"
}


def get_data(dataset: Datasets) -> EventSequence:
    """
    Read a specified dataset and return an EventSequence object.
    """
    if dataset == "radoslaw-email":
        return __read_radoslaw(filename_map[dataset])
    else:
        raise Exception("{} dataset is not available".format(dataset))


def get_keys():
    return filename_map.keys()


def __read_radoslaw(filename: str) -> EventSequence:
    events = []
    with open(filename, "r") as f:
        separator_pat = r"\W+"
        comment_pat = r"^%"
        for row in __read_rows(f, separator_pat, comment_pat):
            events.append(Event(int(row[0]), int(row[1]), float(row[3]), int(row[3])))
    return EventSequence(list_of_events=events)


def __read_rows(
    f: any, separator: str, comment: str
) -> typing.Generator[typing.List[str], None, None]:
    if separator == None:
        raise Exception("separator expression cannot be None")

    separator_pat = re.compile(separator)
    comment_pat = None if comment == None else re.compile(comment)

    for line in f:
        if comment_pat != None and comment_pat.match(line):
            continue

        yield [col.strip() for col in separator_pat.split(line)]
