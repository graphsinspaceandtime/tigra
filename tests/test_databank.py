import sys

sys.path.append("..")

import data.bank as databank

for dataset in databank.get_keys():
    print("dataset:", dataset)
    es = databank.get_data(dataset)

    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
