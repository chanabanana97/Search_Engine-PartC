import os
from spell_checker import Spell_Checker
import time
from thesaurus import Thesaurus
# from search_engine_best import SearchEngine
#
# bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
# search_engine = SearchEngine()
# start = time.time()
# search_engine.build_index_from_parquet(bench_data_path)
# print(time.time()-start)
# print(search_engine.our_data)
# n_relevant, ranked_docs_id = search_engine.search("bioweapon")
# print(n_relevant)
# print("ranked_docs_id")
# print(ranked_docs_id)


query_bad_spelling = ["virus", "you", "are", "tiday", "trump", "coronavirus", "covid-19"]
print(query_bad_spelling)
our_thes = Thesaurus()
returned = our_thes.update(query_bad_spelling)
# our_spell_checker = Spell_Checker()
# returned = our_spell_checker.update(query_bad_spelling)
print(returned)

# test = [1,2,3,4,5,6,7,8,9]
# print(test[:5])