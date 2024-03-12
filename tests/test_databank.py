import sys
import time

sys.path.append("..")

import data.bank as databank

for dataset in databank.get_keys():
    print("dataset:", dataset)
    start_time = time.time()
    es = databank.get_data(dataset)
    end_time = time.time()
    print("Loading time: {}s".format(end_time, start_time))

    # print("num-events=", es.num_events())
    # print("time-span =", es.get_timespan_in_days())
