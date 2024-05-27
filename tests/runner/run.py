import sys
import argparse
import typing
from pathlib import Path

sys.path.append("../..")

from data.bank import Databank, Dataset
from tigra.core.dgraph import DynamicAggregateGraph, DynamicGraph
from tigra.embeddings.sota import ctdne_wrapper
from tigra.embeddings.sota.stwalk import STWalk
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.hypertuner import CTDNE_Tuner, STWalk_Tuner, Dynnode2vec_Tuner


__databank = Databank("/home/shared/tigra/data")

__methods = ["ctdne", "stwalk", "dynnode2vec"]


class Runner:
    def __init__(
        self,
        dataset: str,
        method: str,
        __databank: Databank,
        __methods: typing.List[str],
    ) -> None:
        if dataset not in __databank.get_keys():
            raise Exception(f"dataset {dataset} is not available in dataloader")
        if method not in __methods:
            raise Exception(f"method {method} is not available")
        self.__databank = __databank
        self.dataset = typing.cast(Dataset, dataset)
        self.method = method
        self.__load_dataset()
        self.__create_graph()

    def __load_dataset(self) -> None:
        self.event_sequence = self.__databank.get_data(self.dataset)

    def __create_graph(self) -> None:
        if self.method in ["ctdne"]:
            self.graph = DynamicAggregateGraph(
                f"({self.method}-{self.dataset})", self.event_sequence, directed=False
            )
        else:
            self.graph = DynamicGraph()
            self.graph.init_from_event_sequence(
                self.event_sequence, time_resolution="month", directed=False
            )

    def make_embedding(self) -> None:
        if self.method == "ctdne":
            __graph = typing.cast(DynamicAggregateGraph, self.graph)
            self.embeddings = ctdne_wrapper.Ctdne(10, workers=4)
            self.embeddings.make_embedding(__graph)
        elif self.method == "stwalk":
            __graph = typing.cast(DynamicGraph, self.graph)
            __time_step = len(__graph.get_nx_graph_sequence())
            self.embeddings = STWalk(
                number_restart=40,
                walk_length=10,
                dimension=5,
                time_step=__time_step,
                time_window_size=1,
                workers=4,
                vocab_window_size=5,
            )
            self.embeddings.make_embedding(self.graph)
        elif self.method == "dynnode2vec":
            __graph = typing.cast(DynamicGraph, self.graph)
            self.embeddings = Dynnode2vec(10)
            self.embeddings.make_embedding(__graph)
        else:
            raise Exception(f"unknown method {self.method}")

    def reconstruction_eval(self) -> typing.Tuple[float, float, float]:
        if self.method == "ctdne":
            __graph = typing.cast(DynamicAggregateGraph, self.graph)
            gre = GraphReconstructionEval(
                __graph.get_aggregate_graph(), self.embeddings.embeddings[0]
            )
        elif self.method == "stwalk":
            __graph = typing.cast(DynamicGraph, self.graph)
            self.__last_graph = __graph.get_graph_sequence()[-1]
            __last_embedding = self.embeddings.embeddings[-1]
            gre = GraphReconstructionEval(self.__last_graph, __last_embedding)
        elif self.method == "dynnode2vec":
            __graph = typing.cast(DynamicGraph, self.graph)
            self.__last_graph = __graph.get_graph_sequence()[-1]
            __last_embedding = self.embeddings.embeddings[-1]
            gre = GraphReconstructionEval(self.__last_graph, __last_embedding)
        else:
            raise Exception(f"unknown method {self.method}")
        return (gre.get_avg_precision(), gre.get_avg_recall(), gre.getF1())

    def run_tuner(self) -> None:
        if self.method == "ctdne":
            __graph = typing.cast(DynamicAggregateGraph, self.graph)
            tuner = CTDNE_Tuner(
                __graph.get_aggregate_graph(),
                __graph,
                [0.5, 1.0, 2.0, 4.0],
                [0.5, 1.0, 2.0, 4.0],
                num_runs=5,
            )
        elif self.method == "stwalk":
            tuner = STWalk_Tuner(
                self.__last_graph,
                self.graph,
                [1, 3, 5],
                num_runs=5,
                n_walks_per_node=40,
            )
        elif self.method == "dynnode2vec":
            tuner = Dynnode2vec_Tuner(
                self.__last_graph,
                self.graph,
                [0.5, 1.0, 2.0, 4.0],
                [0.5, 1.0, 2.0, 4.0],
                num_runs=10,
                parallel_processes=16,
                walk_length=64,
                n_walks_per_node=10,
            )
        else:
            raise Exception(f"unknown method {self.method}")
        tuner.tune()
        tuner.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tuner", description="Hyperparameter tuner cli"
    )

    parser.add_argument("-d", "--dataset", type=str, nargs="*")
    parser.add_argument("-m", "--method", type=str, nargs="*")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-methods", action="store_true")

    args = parser.parse_args()

    if args.list_datasets:
        print(__databank.get_keys())
    elif args.list_methods:
        print(__methods)
    else:
        datasets = args.dataset if args.dataset != None else __databank.get_keys()
        methods = args.methods if args.method != None else __methods
        combos = []
        for dataset in datasets:
            for method in methods:
                combos.append((dataset, method))
        runners = map(lambda t: Runner(t[0], t[1], __databank, __methods), combos)
        for runner in runners:
            print(f"Running for dataset: {runner.dataset} and method: {runner.method}")
            try:
                runner.make_embedding()
                print(runner.reconstruction_eval())
                runner.run_tuner()
            except:
                print("Something went wrong")
