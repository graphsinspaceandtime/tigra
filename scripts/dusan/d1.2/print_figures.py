import sys

sys.path.append("../../..")

from data.bank import Databank

db = Databank()


def basic_props(dataset):
    print("\\begin{figure}")
    print("    \\centering")
    print("    \\caption{" + dataset + "}")
    print("    \\label{fig:data-props-" + dataset + "}")
    print(
        "    \\includegraphics[width=.9\\textwidth]{graphics/data/" + dataset + ".png}"
    )
    print("\\end{figure}")


for i, dataset in enumerate(db.get_keys()):
    basic_props(dataset)
    print()
