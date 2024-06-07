import sys

sys.path.append("..")

from data.bank import Databank

databank = Databank()

for dataset in databank.get_keys():
    print("dataset:", dataset)
    es = databank.get_data(dataset)

    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
