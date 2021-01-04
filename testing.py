import os
from spell_checker import spell_checker
import time
from search_engine_best import SearchEngine

bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
search_engine = SearchEngine()
start = time.time()
search_engine.build_index_from_parquet(bench_data_path)
print(time.time()-start)
print(search_engine.our_data)
n_relevant, ranked_docs_id = search_engine.search("bioweapon")
print(n_relevant)
print("ranked_docs_id")
print(ranked_docs_id)


query_bad_spelling = ["howe", "aree", "ypu", "tiday"]
print(query_bad_spelling)
our_spell_checker = spell_checker()
returned = our_spell_checker.correct_spelling(query_bad_spelling)
print(returned)