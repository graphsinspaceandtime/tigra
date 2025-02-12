"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module for tuning hyperparameters of dynamic graph embedding methods
"""

from abc import ABC, abstractmethod
import typing

from statistics import mean
import tqdm
import math

from sklearn.model_selection import ParameterGrid

from tigra.embeddings.embedding import Embedding
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.utils import n_nodes_to_ae_layers

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from tigra.embeddings.sota.stwalk import STWalk
from tigra.embeddings.sota.dynae_wrapper import DynAE
from tigra.embeddings.sota.dynrnn_wrapper import DynRNN
from tigra.embeddings.sota.dynaernn_wrapper import DynAERNN
from tigra.embeddings.sota.dyngem_wrapper import DynGEM

from tigra.embeddings.rndw.rndwbase import DynN2V

from tigra.embeddings.rndw.lidrw import (
    DynN2V_LIDDelta,
    DynN2V_LIDRW,
    DynN2V_LIDRWDelta
)


class Tuner(ABC):
    """
    Base class for hyper-parameter tuning and intrinsic graph embedding evaluation
    """

    def __init__(self, graph, data, params_desc, num_runs=1):
        """
        graph       -- referent graph used when reconstructing graph from embedding
        data        -- data used to construct embedding (sequence of events or dynamic aggregate graph)
        params_desc -- description of hyperparams (string)
        num_runs    -- the number of runs for each configuration
        """
        self.graph = graph
        self.data = data
        self.params_desc = params_desc
        self.num_runs = num_runs

    @abstractmethod
    def create_instance(self, conf_index) -> Embedding:
        """
        This method has to create an embedding object for the configuration at conf_index
        """
        pass

    @abstractmethod
    def create_configurations(self) -> typing.List:
        """
        This method has to create and return a list of configurations
        """
        pass

    def tune(self, progress_bar=False):
        self.best = None
        self.eval_res = []
        self.confs = self.create_configurations()

        iterator = range(len(self.confs))
        if progress_bar:
            iterator = tqdm.tqdm(iterator)

        for i in iterator:
            prec_values, rec_values, f1_values = [], [], []
            for run_num in range(self.num_runs):
                try:
                    current_instance = self.create_instance(i)
                except:
                    print("[tuner] failed to create instance")
                    print("[tuner] configuration: " + str(self.confs[i]))
                    print("[tuner] run_num: " + str(run_num))
                    continue
                current_instance.make_embedding(self.data)
                last_embedding = current_instance.embeddings[-1]
                if last_embedding == None:
                    raise Exception(
                        "this can't happen (last_embedding in abstract Tuner class is None)"
                    )
                gre = GraphReconstructionEval(
                    self.graph, last_embedding, report_warnings=False
                )
                prec = gre.get_avg_precision()
                rec = gre.get_avg_recall()
                f1 = gre.getF1()
                prec_values.append(prec)
                rec_values.append(rec)
                f1_values.append(f1)

            mean_prec = mean(prec_values)
            mean_rec = mean(rec_values)
            mean_f1 = mean(f1_values)

            self.eval_res.append((i, mean_f1, mean_prec, mean_rec))
            if self.best == None or mean_f1 > self.best[1]:
                self.best = (i, mean_f1, mean_prec, mean_rec)

            if not progress_bar:
                print("Done ", i, " out of", len(self.confs))

    def print_report(self):
        print(self.params_desc + ",F1,PREC,REC")
        for i in range(len(self.confs)):
            s = ",".join([str(x) for x in self.confs[i]])
            eval = self.eval_res[i]
            s += "," + str(eval[1]) + "," + str(eval[2]) + "," + str(eval[3])
            print(s)

        print("Best conf: ", self.best, " [f1, prec, rec]")
        if self.best != None:
            print("Best conf params: ", self.confs[self.best[0]])


EMBEDDING_DIMENSIONS = [10, 25, 50, 100, 200]
# EMBEDDING_DIMENSIONS = [10, 25]


class N2VMethods_Tuner(Tuner):
    """
    Base tuner class for graph embedding methods based on random walks
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walks_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(graph, data, "DIM,P,Q", num_runs=num_runs)
        self.p_values = p_values
        self.q_values = q_values
        self.walk_length_values = walk_length_values
        self.num_walks_values = num_walks_values
        self.parallel_processes = parallel_processes

    def create_configurations(self):
        confs = []
        for d in EMBEDDING_DIMENSIONS:
            for p in self.p_values:
                for q in self.q_values:
                    for wl in self.walk_length_values:
                        for nw in self.num_walks_values:
                            confs.append((d, p, q, wl, nw))

        return confs


class Dynnode2vec_Tuner(N2VMethods_Tuner):
    """
    Tuner for Dynnode2vec
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walk_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walk_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return Dynnode2vec(
            dim,
            p=p,
            q=q,
            walk_length=wl,
            n_walks_per_node=nw,
            parallel_processes=self.parallel_processes,
        )


class DN2V_Tuner(N2VMethods_Tuner):
    """
    Tuner for Dynnode2vec (alternative implementation)
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walk_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walk_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return DynN2V(
            dim, p, q, num_walks=nw, walk_length=wl, num_workers=self.parallel_processes
        )


class CTDNE_Tuner(N2VMethods_Tuner):
    """
    Tuner for CTDNE
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walks_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walks_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return Ctdne(
            dim,
            p,
            q,
            walk_length=wl,
            num_walks=nw,
            workers=self.parallel_processes,
        )


class STWalk_Tuner(Tuner):
    """
    Tuner for STWalk
    """

    def __init__(
        self,
        graph,
        data,
        time_window_values,
        num_runs=1,
        walk_length=10,
        n_walks_per_node=10,
        parallel_processes=4,
    ):
        super().__init__(graph, data, "DIM,TIME_WINDOW", num_runs=num_runs)
        self.time_window_values = time_window_values
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.parallel_processes = parallel_processes

    def create_configurations(self):
        confs = []
        for d in EMBEDDING_DIMENSIONS:
            for tw in self.time_window_values:
                confs.append((d, tw))

        return confs

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        tws = self.confs[conf_index][1]
        ts = len(self.data.get_nx_graph_sequence())
        return STWalk(
            number_restart=self.n_walks_per_node,
            walk_length=self.walk_length,
            dimension=dim,
            time_step=ts,
            time_window_size=tws,
            workers=self.parallel_processes,
            vocab_window_size=5,
        )


class DynamicEmbedding_Tuner(Tuner):
    """
    Tuner for DynamicEmbedding methods (DynGem, DynAE, DynRNN, DynAERNN)
    """

    ParamGrid = typing.Mapping[str, typing.Sequence[typing.Any]]
    Method = typing.Literal["DynAE", "DynRNN", "DynAERNN", "DynGEM"]

    method: Method
    param_grid: ParamGrid
    epochs: int
    batch_size: int
    n_nodes: int
    has_cuda: bool
    pick_n_layers_from_middle: int

    def __init__(
        self,
        graph,
        data,
        params_desc,
        method: Method,
        param_grid: ParamGrid,
        epochs: int,
        batch_size: int,
        n_nodes: int,
        has_cuda: bool,
        num_runs=1,
        pick_n_layers_from_middle=0,
    ):
        super().__init__(graph, data, params_desc, num_runs)
        self.method = method
        self.param_grid = param_grid
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.has_cuda = has_cuda
        self.pick_n_layers_from_middle = pick_n_layers_from_middle

    def create_configurations(self) -> typing.List:
        return list(ParameterGrid(self.param_grid))

    def create_instance(self, conf_index) -> Embedding:
        conf = self.confs[conf_index]
        fixed_params = {
            "beta": 5,
            "nu1": 1e-6,
            "nu2": 1e-6,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        }

        n_units = n_nodes_to_ae_layers(self.n_nodes, conf["dim"])
        if self.pick_n_layers_from_middle != 0:
            if self.pick_n_layers_from_middle > len(n_units):
                print(
                    "[tuner] DynamicEmbedding_Tuner instance field 'pick_n_layers_from_middle' larger than length of 'n_units'"
                )
                raise Exception(
                    "field 'pick_n_layers_from_middle' larger than length of 'n_units'"
                )
            padding = (len(n_units) - self.pick_n_layers_from_middle) / 2
            padding_left = math.ceil(padding)
            padding_right = math.floor(padding)
            n_units = n_units[padding_left:padding_right]

        if self.method == "DynAE":
            return DynAE(
                n_units=n_units, has_cuda=self.has_cuda, **conf, **fixed_params
            )
        if self.method == "DynRNN":
            return DynRNN(
                n_units=n_units, has_cuda=self.has_cuda, **conf, **fixed_params
            )
        if self.method == "DynAERNN":
            return DynAERNN(
                ae_units=n_units,
                has_cuda=self.has_cuda,
                rnn_units=n_units,
                **conf,
                **fixed_params,
            )
        if self.method == "DynGEM":
            return DynGEM(
                n_units=n_units, has_cuda=self.has_cuda, **conf, **fixed_params
            )
        raise Exception(f"method '{self.method}' is incorrect")


class DynN2V_LIDDelta_Tuner(N2VMethods_Tuner):
    """
    Tuner for DynN2V_LIDDelta
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walk_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walk_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return DynN2V_LIDDelta(
            dim, p, q, num_walks=nw, walk_length=wl, num_workers=self.parallel_processes
        )
    

class DynN2V_LIDRW_Tuner(N2VMethods_Tuner):
    """
    Tuner for DynN2V_LIDRW
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walk_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walk_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return DynN2V_LIDRW(
            dim, p, q, num_walks=nw, walk_length=wl, num_workers=self.parallel_processes
        )
    

class DynN2V_LIDRWDelta_Tuner(N2VMethods_Tuner):
    """
    Tuner for DynN2V_LIDRWDelta
    """

    def __init__(
        self,
        graph,
        data,
        p_values,
        q_values,
        walk_length_values,
        num_walk_values,
        num_runs=1,
        parallel_processes=4,
    ):
        super().__init__(
            graph,
            data,
            p_values,
            q_values,
            walk_length_values,
            num_walk_values,
            num_runs=num_runs,
            parallel_processes=parallel_processes,
        )

    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        wl = self.confs[conf_index][3]
        nw = self.confs[conf_index][4]
        return DynN2V_LIDRWDelta(
            dim, p, q, num_walks=nw, walk_length=wl, num_workers=self.parallel_processes
        )