if __name__ == '__main__':
    import os
    import sys
    import re
    from datetime import datetime
    import pandas as pd
    import pyarrow.parquet as pq
    import time
    import timeit
    import importlib
    import logging

    logging.basicConfig(filename='part_c_tests.log', level=logging.DEBUG,
                        filemode='w', format='%(levelname)s %(asctime)s: %(message)s')
    import metrics


    def test_file_exists(fn):
        if os.path.exists(fn):
            return True
        logging.error(f'{fn} does not exist.')
        return False


    tid_ptrn = re.compile('\d+')


    def invalid_tweet_id(tid):
        if not isinstance(tid, str):
            tid = str(tid)
        if tid_ptrn.fullmatch(tid) is None:
            return True
        return False


    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')
    model_dir = os.path.join('.', 'model')

    start = datetime.now()
    try:
        # is the report there?
        test_file_exists('report_part_c.docx')
        # is benchmark data under 'data' folder?
        bench_lbls = None
        q2n_relevant = None
        if not test_file_exists(bench_data_path) or \
                not test_file_exists(bench_lbls_path):
            logging.error("Benchmark data does exist under the 'data' folder.")
            sys.exit(-1)
        else:
            bench_lbls = pd.read_csv(bench_lbls_path,
                                     dtype={'query': int, 'tweet': str, 'y_true': int})
            q2n_relevant = bench_lbls.groupby('query')['y_true'].sum().to_dict()
            logging.info("Successfully loaded benchmark labels data.")

        # is queries file under data?
        queries = None
        if not test_file_exists(queries_path):
            logging.error("Queries data not found ~> skipping some tests.")
        else:
            queries = pd.read_csv(os.path.join('data', 'queries_train.tsv'), sep='\t')
            logging.info("Successfully loaded queries data.")

        import configuration

        config = configuration.ConfigClass()

        # do we need to download a pretrained model?
        model_url = config.get_model_url()
        if model_url is not None and config.get_download_model():
            import utils

            dest_path = 'model.zip'
            utils.download_file_from_google_drive(model_url, dest_path)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            if os.path.exists(dest_path):
                utils.unzip_file(dest_path, model_dir)
                logging.info(f'Successfully downloaded and extracted pretrained model into {model_dir}.')
            else:
                logging.error('model.zip file does not exists.')

        # test for each search engine module
        # engine_modules = ['search_engine_' + name for name in ['1', '2', 'best']]
        # engine_modules = ['search_engine_' + name for name in ['1', '2', '3', '4', 'best']]
        engine_modules = ['search_engine_1']
        for engine_module in engine_modules:
            try:
                # does the module file exist?
                if not test_file_exists(engine_module + '.py'):
                    continue
                # try importing the module
                se = importlib.import_module(engine_module)
                logging.info(f"Successfully imported module {engine_module}.")
                engine = se.SearchEngine(config=config)

                # test building an index and doing so in <1 minute
                build_idx_time = timeit.timeit(
                    "engine.build_index_from_parquet(bench_data_path)",
                    globals=globals(), number=1
                )
                logging.debug(
                    f"Building the index in {engine_module} for benchmark data took {build_idx_time} seconds.")
                if build_idx_time > 60:
                    logging.error('Parsing and index our *small* benchmark dataset took over a minute!')
                # test loading precomputed model
                engine.load_precomputed_model(model_dir)

                # test that we can run one query and get results in the format we expect
                n_res, res = engine.search('bioweapon')
                if n_res is None or res is None or n_res < 1 or len(res) < 1:
                    logging.error('basic query for the word bioweapon returned no results')
                else:
                    logging.debug(f"{engine_module} successfully returned {n_res} results for the query 'bioweapon'.")
                    invalid_tweet_ids = [doc_id for doc_id in res if invalid_tweet_id(doc_id)]
                    if len(invalid_tweet_ids) > 0:
                        logging.error("the query 'bioweapon' returned results that are not valid tweet ids: " + str(
                            invalid_tweet_ids[:10]))

                # run multiple queries and test that no query takes > 10 seconds
                queries_results = []
                if queries is not None:
                    for i, row in queries.iterrows():
                        q_id = row['query_id']
                        q_keywords = row['keywords']
                        start_time = time.time()
                        q_n_res, q_res = engine.search(q_keywords)
                        end_time = time.time()
                        q_time = end_time - start_time
                        if q_n_res is None or q_res is None or q_n_res < 1 or len(q_res) < 1:
                            logging.error(f"Query {q_id} with keywords '{q_keywords}' returned no results.")
                        else:
                            logging.debug(
                                f"{engine_module} successfully returned {q_n_res} results for query number {q_id}.")
                            invalid_tweet_ids = [doc_id for doc_id in q_res if invalid_tweet_id(doc_id)]
                            if len(invalid_tweet_ids) > 0:
                                logging.error(f"Query  {q_id} returned results that are not valid tweet ids: " + str(
                                    invalid_tweet_ids[:10]))
                            queries_results.extend(
                                [(q_id, str(doc_id)) for doc_id in q_res if not invalid_tweet_id(doc_id)])
                        if q_time > 10:
                            logging.error(f"Query {q_id} with keywords '{q_keywords}' took more than 10 seconds.")
                queries_results = pd.DataFrame(queries_results, columns=['query', 'tweet'])

                # merge query results with labels benchmark
                q_results_labeled = None
                if bench_lbls is not None and len(queries_results) > 0:
                    q_results_labeled = pd.merge(queries_results, bench_lbls,
                                                 on=['query', 'tweet'], how='inner', suffixes=('_result', '_bench'))
                    # q_results_labeled.rename(columns={'y_true': 'label'})
                    zero_recall_qs = [q_id for q_id, rel in q2n_relevant.items() \
                                      if metrics.recall_single(q_results_labeled, rel, q_id) == 0]
                    if len(zero_recall_qs) > 0:
                        logging.warning(
                            f"{engine_module}'s recall for the following queries was zero {zero_recall_qs}.")

                if q_results_labeled is not None:
                    # test that MAP > 0
                    results_map = metrics.map(q_results_labeled)
                    logging.debug(f"{engine_module} results have MAP value of {results_map}.")
                    if results_map <= 0 or results_map > 1:
                        logging.error(f'{engine_module} results MAP value is out of range (0,1).')

                    # test that the average across queries of precision,
                    # precision@5, precision@10, precision@50, and recall
                    # is in [0,1].
                    prec, p5, p10, p50, recall = \
                        metrics.precision(q_results_labeled), \
                        metrics.precision(q_results_labeled.groupby('query').head(5)), \
                        metrics.precision(q_results_labeled.groupby('query').head(10)), \
                        metrics.precision(q_results_labeled.groupby('query').head(50)), \
                        metrics.recall(q_results_labeled, q2n_relevant)
                    # print("query 1")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 1))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 1))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 1))
                    # print("query 2")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 2))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 2))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 2))
                    # print("query 3")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 3))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 3))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 3))
                    # print("query 4")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 4))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 4))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 4))
                    # print("query 5")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 5))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 5))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 5))
                    # print("query 6")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 6))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 6))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 6))
                    # print("query 7")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 7))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 7))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 7))
                    # print("query 8")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 8))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 8))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 8))
                    # print("query 9")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 9))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 9))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 9))
                    # print("query 10")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 10))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 10))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 10))
                    # print("query 11")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 11))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 11))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 11))
                    # print("query 12")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 12))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 12))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 12))
                    # print("query 13")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 13))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 13))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 13))
                    # print("query 14")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 14))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 14))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 14))
                    # print("query 15")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 15))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 15))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 15))
                    # print("query 16")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 16))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 16))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 16))
                    # print("query 17")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 17))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 17))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 17))
                    # print("query 18")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 18))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 18))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 18))
                    # print("query 19")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 19))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 19))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 19))
                    # print("query 20")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 20))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 20))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 20))
                    # print("query 21")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 21))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 21))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 21))
                    # print("query 22")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 22))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 22))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 22))
                    # print("query 23")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 23))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 23))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 23))
                    # print("query 24")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 24))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 24))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 24))
                    # print("query 25")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 25))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 25))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 25))
                    # print("query 26")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 26))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 26))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 26))
                    # print("query 27")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 27))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 27))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 27))
                    # print("query 28")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 28))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 28))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 28))
                    # print("query 29")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 29))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 29))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 29))
                    # print("query 30")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 30))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 30))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 30))
                    # print("query 31")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 31))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 31))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 31))
                    # print("query 32")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 32))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 32))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 32))
                    # print("query 33")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 33))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 33))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 33))
                    # print("query 34")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 34))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 34))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 34))
                    # print("query 35")
                    # print(metrics.precision(q_results_labeled))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(5), True, 35))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(10), True, 35))
                    # print(metrics.precision(q_results_labeled.groupby('query').head(50), True, 35))

                    logging.debug(f"{engine_module} results produced average precision of {prec}.")
                    logging.debug(f"{engine_module} results produced average precision@5 of {p5}.")
                    logging.debug(f"{engine_module} results produced average precision@10 of {p10}.")
                    logging.debug(f"{engine_module} results produced average precision@50 of {p50}.")
                    logging.debug(f"{engine_module} results produced average recall of {recall}.")
                    if prec < 0 or prec > 1:
                        logging.error(f"The average precision for {engine_module} is out of range [0,1].")
                    if p5 < 0 or p5 > 1:
                        logging.error(f"The average precision@5 for {engine_module} is out of range [0,1].")
                    if p5 < 0 or p5 > 1:
                        logging.error(f"The average precision@5 for {engine_module} is out of range [0,1].")
                    if p50 < 0 or p50 > 1:
                        logging.error(f"The average precision@50 for {engine_module} is out of range [0,1].")
                    if recall < 0 or recall > 1:
                        logging.error(f"The average recall for {engine_module} is out of range [0,1].")
                    print(f'map:{results_map}\nrecall:{recall}') # TODO delete
                if engine_module == 'search_engine_best' and \
                        test_file_exists('idx_bench.pkl'):
                    logging.info('idx_bench.pkl found!')
                    engine.load_index('idx_bench.pkl')
                    logging.info('Successfully loaded idx_bench.pkl using search_engine_best.')

            except Exception as e:
                logging.error(f'The following error occured while testing the module {engine_module}.')
                logging.error(e, exc_info=True)

    except Exception as e:
        logging.error(e, exc_info=True)

    run_time = datetime.now() - start
    logging.debug(f'Total runtime was: {run_time}')