import typing
from tigra.core.dgraph import EventSequence
from data import bank


def choose_graph(
    base_path: typing.Union[str, None] = None,
    name: typing.Union[bank.Dataset, None] = None,
    shrink_sequence=False,
) -> typing.Tuple[EventSequence, bank.DatasetMetadata]:
    db = bank.Databank(path=base_path)

    if name != None:
        return (db.get_data(name, shrink_sequence), db.get_metadata(name))

    print("Chose dataset:")
    keys = db.get_keys()
    for i, set_name in enumerate(keys):
        print(f"    {i}) {set_name}")

    print()
    chosen_key = int(input("> "))

    return (
        db.get_data(keys[chosen_key], shrink_sequence),
        db.get_metadata(keys[chosen_key]),
    )
