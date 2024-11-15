import bz2
import json
import os

OPENAI_QUERIES_FILENAME: str = "openai_queries.json.bz2"
DEEP1B_QUERIES_FILENAME: str = "deep1b_queries.json"
SO_QUERIES_FILENAME: str = "so_queries.json"


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


def register(registry):
    registry.register_param_source("knn-param-source", KnnParamSource)
