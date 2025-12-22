from app import run_analysis
import pprint
r = run_analysis()
print('keys:')
pprint.pprint(list(r.keys()))
print('\ntypes:')
ptype = {k: type(v).__name__ for k,v in r.items()}
pprint.pprint(ptype)
print('\nstats_results:')
pprint.pprint(r.get('stats_results'))
print('\nmodel_performance:')
pprint.pprint(r.get('model_performance'))
