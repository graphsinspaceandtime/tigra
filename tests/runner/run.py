import sys
import argparse
import typing
from pathlib import Path
from abc import ABC, abstractmethod

sys.path.append("../..")

from data.bank import Databank, Dataset
from tigra.core.dgraph import DynamicAggregateGraph, DynamicGraph, EventSequence
from tigra.embeddings.sota import ctdne_wrapper
from tigra.embeddings.sota.stwalk import STWalk
from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.hypertuner import CTDNE_Tuner, STWalk_Tuner, Dynnode2vec_Tuner


__databank = Databank("/home/shared/tigra/data")

__methods = ["ctdne", "stwalk", "dynnode2vec"]


class MethodRunner(ABC):
    @abstractmethod
    def reconstruction_eval(self) -> GraphReconstructionEval:
        pass

    @abstractmethod
    def tuner_eval(self) -> None:
        pass


class CTDNERunner(MethodRunner):
    def __init__(self, graph: typing.Any) -> None:
        self.graph = typing.cast(DynamicAggregateGraph, graph)

    def reconstruction_eval(self) -> GraphReconstructionEval:
        __graph = typing.cast(DynamicAggregateGraph, self.graph)
        __embeddings = ctdne_wrapper.Ctdne(10, workers=4)
        __embeddings.make_embedding(__graph)
        gre = GraphReconstructionEval(
            __graph.get_aggregate_graph(), __embeddings.embeddings[0]
        )
        return gre

    def tuner_eval(self) -> None:
        tuner = CTDNE_Tuner(
            self.graph.get_aggregate_graph(),
            self.graph,
            [0.5, 1.0, 2.0, 4.0],
            [0.5, 1.0, 2.0, 4.0],
            num_runs=5,
        )
        tuner.tune()
        tuner.print_report()


class STWalkRunner(MethodRunner):
    def __init__(self, graph: typing.Any) -> None:
        self.graph = typing.cast(DynamicGraph, graph)

    def reconstruction_eval(self) -> GraphReconstructionEval:
        __time_step = len(self.graph.get_nx_graph_sequence())
        __embeddings = STWalk(
            number_restart=40,
            walk_length=10,
            dimension=5,
            time_step=__time_step,
            time_window_size=1,
            workers=4,
            vocab_window_size=5,
        )
        __embeddings.make_embedding(self.graph)
        self.__last_graph = self.graph.get_graph_sequence()[-1]
        __last_embedding = __embeddings.embeddings[-1]
        gre = GraphReconstructionEval(self.__last_graph, __last_embedding)
        return gre

    def tuner_eval(self) -> None:
        tuner = STWalk_Tuner(
            self.__last_graph,
            self.graph,
            [1, 3, 5],
            num_runs=5,
            n_walks_per_node=40,
        )
        tuner.tune()
        tuner.print_report()


class DynNode2VecRunner(MethodRunner):
    def __init__(self, graph: typing.Any) -> None:
        self.graph = typing.cast(DynamicGraph, graph)

    def reconstruction_eval(self) -> GraphReconstructionEval:
        __embeddings = Dynnode2vec(10)
        __embeddings.make_embedding(self.graph)
        self.__last_graph = self.graph.get_graph_sequence()[-1]
        __last_embedding = __embeddings.embeddings[-1]
        gre = GraphReconstructionEval(self.__last_graph, __last_embedding)
        return gre

    def tuner_eval(self) -> None:
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
        tuner.tune()
        tuner.print_report()


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
        es = self.__databank.get_data(self.dataset)
        graph = self.__create_graph(es)
        self.method_runner = self.__create_method_runner(graph, method)

    def __create_graph(self, es: EventSequence) -> typing.Any:
        if self.method in ["ctdne"]:
            graph = DynamicAggregateGraph(
                f"({self.method}-{self.dataset})", es, directed=False
            )
        else:
            graph = DynamicGraph()
            graph.init_from_event_sequence(es, time_resolution="month", directed=False)
        return graph

    def __create_method_runner(self, graph: typing.Any, method: str) -> MethodRunner:
        if method == "ctdne":
            return CTDNERunner(graph)
        elif method == "stwalk":
            return STWalkRunner(graph)
        elif method == "dynnode2vec":
            return DynNode2VecRunner(graph)
        else:
            raise Exception(f"unknown method {method}")

    def reconstruction_eval(self) -> typing.Tuple[float, float, float]:
        gre = self.method_runner.reconstruction_eval()
        return (gre.get_avg_precision(), gre.get_avg_recall(), gre.getF1())

    def tuner_eval(self) -> None:
        self.method_runner.tuner_eval()


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
                print(runner.reconstruction_eval())
                runner.tuner_eval()
            except:
                print("Something went wrong")
