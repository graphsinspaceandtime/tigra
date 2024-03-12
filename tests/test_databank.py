import sys
import time

sys.path.append("..")

import data.bank as databank

# print(",".join(databank.get_keys()))
results = []

for dataset in databank.get_keys():
    # print("dataset:", dataset)
    start_time = time.time()
    es = databank.get_data(dataset)
    end_time = time.time()
    # print("Loading time: {}s".format(end_time - start_time))
    results.append(str(end_time - start_time))

print(",".join(results))

# print("num-events=", es.num_events())
# print("time-span =", es.get_timespan_in_days())
