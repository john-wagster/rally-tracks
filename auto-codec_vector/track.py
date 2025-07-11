import bz2
import json
import os
from collections import defaultdict
from typing import Dict


Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, Dict[str, float]]

OPENAI_QUERIES_FILENAME: str = "openai_queries.json.bz2"
DEEP1B_QUERIES_FILENAME: str = "deep1b_queries.json"
SO_QUERIES_FILENAME: str = "so_queries.json"

def calc_ndcg(qrels: Qrels, results: Results, k_list: list):
    import pytrec_eval as pe

    for qid, rels in results.items():
        for pid in list(rels):
            if qid == pid:
                results[qid].pop(pid)

    scores = defaultdict(float)

    metrics = ["ndcg_cut"]
    pytrec_strings = {f"{metric}.{','.join([str(k) for k in k_list])}" for metric in metrics}
    evaluator = pe.RelevanceEvaluator(qrels, pytrec_strings)
    pytrec_scores = evaluator.evaluate(results)

    for query_id in pytrec_scores.keys():
        for metric in metrics:
            for k in k_list:
                scores[f"{metric}@{k}"] += pytrec_scores[query_id][f"{metric}_{k}"]

    queries_count = len(pytrec_scores.keys())
    if queries_count == 0:
        return scores

    for metric in metrics:
        for k in k_list:
            scores[f"{metric}@{k}"] = float(scores[f"{metric}@{k}"] / queries_count)

    return scores


class KnnParamSource:
    def __init__(self, track, params, **kwargs):
        # choose a suitable index: if there is only one defined for this track
        # choose that one, but let the user always override index
        if len(track.indices) == 1:
            default_index = track.indices[0].name
        else:
            default_index = "_all"

        self._index_name = params.get("index", default_index)
        self._cache = params.get("cache", False)
        self._params = params
        self._queries = []

        raw_queries_file = self._params.get("queries_file", OPENAI_QUERIES_FILENAME)

        cwd = os.path.dirname(__file__)
        if raw_queries_file.endswith("bz2"):
            handle = bz2.open(os.path.join(cwd, raw_queries_file), "r")
        else:
            handle = open(os.path.join(cwd, raw_queries_file), "r")

        with handle as queries_file:
            self._queries = [json.loads(query) for query in queries_file]

        self._iters = 0
        self._maxIters = len(self._queries)
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        result = {"index": self._index_name, "cache": self._params.get("cache", False), "size": self._params.get("k", 10)}

        result["body"] = {
            "knn": {
                "field": self._params.get("field-name", "emb"),
                "query_vector": self._queries[self._iters],
                "k": self._params.get("k", 10),
                "num_candidates": self._params.get("num-candidates", 50),
            },
            "_source": False,
        }
        if "filter" in self._params:
            result["body"]["knn"]["filter"] = self._params["filter"]

        self._iters += 1
        if self._iters >= self._maxIters:
            self._iters = 0
        return result


class WeightedRecallParamSource:
    def __init__(self, track, params, **kwargs):
        if len(track.indices) == 1:
            default_index = track.indices[0].name
        else:
            default_index = "_all"

        self._query_file = params.get("queries-file", "queries-small.json")
        self._qrels_file = params.get("qrels-file", "qrels-small.tsv")
        self._index_name = params.get("index", default_index)
        self._cache = params.get("cache", False)
        self._k = params.get("k", 10)
        self._num_candidates = params.get("num-candidates", 100)
        self._params = params
        self._queries = []
        self._field_name = params.get("field-name")
        self.infinite = True

        cwd = os.path.dirname(__file__)
        with open(os.path.join(cwd, self._query_file), "r") as file:
            self._queries = json.load(file)
        self._qrels = read_qrels(os.path.join(cwd, self._qrels_file))

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        return {
            "index": self._index_name,
            "cache": self._cache,
            "k": self._k,
            "num_candidates": self._num_candidates,
            "queries": self._queries,
            "qrels": self._qrels,
            "field_name": self._field_name,
        }


# For each query this will generate the weighted terms query, a pruned version and a rescored pruned version of the same query.
# These queries can then be executed and compared for accuracy.
class WeightedTermsRecallRunner:
    async def __call__(self, es, params):
        recall_total = 0
        recall_with_rescore_total = 0
        exact_total = 0
        min_recall = params["k"]
        weighted_term_results = defaultdict(dict)
        pruned_results = defaultdict(dict)
        pruned_rescored_results = defaultdict(dict)

        for query in params["queries"]:
            query_id = query["id"]

            # Build and execute all three queries
            weighted_terms_result = await es.search(
                body=generate_weighted_terms_query(params["field_name"], query[params["field_name"]], 1),
                index=params["index"],
                request_cache=params["cache"],
                size=params["k"],
            )
            pruned_result = await es.search(
                body=generate_pruned_query(params["field_name"], query[params["field_name"]], 1),
                index=params["index"],
                request_cache=params["cache"],
                size=params["k"],
            )
            pruned_rescored_result = await es.search(
                body=generate_rescored_pruned_query(
                    params["field_name"], query[params["field_name"]], params["num_candidates"], 1
                ),
                index=params["index"],
                request_cache=params["cache"],
                size=params["k"],
            )

            weighted_terms_hits = {hit["_source"]["id"]: hit["_score"] for hit in weighted_terms_result["hits"]["hits"]}
            pruned_hits = {hit["_source"]["id"]: hit["_score"] for hit in pruned_result["hits"]["hits"]}
            pruned_rescored_hits = {hit["_source"]["id"]: hit["_score"] for hit in pruned_rescored_result["hits"]["hits"]}

            # Recall calculations as compared to the control/non-pruned hits
            weighted_terms_ids = set(weighted_terms_hits.keys())
            pruned_ids = set(pruned_hits.keys())
            pruned_rescored_ids = set(pruned_rescored_hits.keys())
            current_recall_with_rescore = len(weighted_terms_ids.intersection(pruned_rescored_ids))
            current_recall = len(weighted_terms_ids.intersection(pruned_ids))
            recall_with_rescore_total += current_recall_with_rescore
            recall_total += current_recall
            exact_total += len(weighted_terms_ids)
            min_recall = min(min_recall, current_recall)

            # Construct input to NDCG calculation based on returned hits
            for doc_id, score in weighted_terms_hits.items():
                weighted_term_results[query_id][doc_id] = score
            for doc_id, score in pruned_hits.items():
                pruned_results[query_id][doc_id] = score
            for doc_id, score in pruned_rescored_hits.items():
                pruned_rescored_results[query_id][doc_id] = score

        control_relevance = calc_ndcg(params["qrels"], weighted_term_results, [10, 100])
        pruned_relevance = calc_ndcg(params["qrels"], pruned_results, [10, 100])
        pruned_rescored_relevance = calc_ndcg(params["qrels"], pruned_rescored_results, [10, 100])

        return (
            {
                "avg_recall": float(recall_with_rescore_total / exact_total),  # Calculated on pruned/rescored hits
                "avg_recall_without_rescore": float(recall_total / exact_total),  # Calculated on pruned hits without rescore
                "min_recall": min_recall,  # Calculated on pruned/rescored hits
                "top_k": params["k"],
                "num_candidates": params["num_candidates"],
                "control_ndcg_10": control_relevance["ndcg_cut@10"],
                "control_ndcg_100": control_relevance["ndcg_cut@100"],
                "pruned_ndcg_10": pruned_relevance["ndcg_cut@10"],
                "pruned_ndcg_100": pruned_relevance["ndcg_cut@100"],
                "pruned_rescored_ndcg_10": pruned_rescored_relevance["ndcg_cut@10"],
                "pruned_rescored_ndcg_100": pruned_rescored_relevance["ndcg_cut@100"],
            }
            if exact_total > 0
            else None
        )

    def __repr__(self, *args, **kwargs):
        return "weighted_terms_recall"


def register(registry):
    registry.register_param_source("knn-param-source", KnnParamSource)
    registry.register_param_source("recall-param-source", WeightedRecallParamSource)
    registry.register_runner("recall", WeightedTermsRecallRunner(), async_runner=True)
