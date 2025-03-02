from tigra.data.loaders import RadoslawEmail, CollegeMsg, SyntheticDBA

loader = RadoslawEmail(base_path="../../data")
loader.download()

es = loader.get_event_sequence(shrink=True)
print(es.num_events())

loader = CollegeMsg(base_path="../../data")
loader.download()

es = loader.get_event_sequence(shrink=True)
print(es.num_events())

loader = SyntheticDBA(base_path="../../data")
loader.download()

es = loader.get_event_sequence(shrink=True)
print(es.num_events())
