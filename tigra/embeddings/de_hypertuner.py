"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module for tuning hyperparameters of dynamic graph embedding methods

Dynamic Embedding method tuners
"""

from abc import ABC, abstractmethod
import typing

from statistics import mean
import tqdm
import math
import json

from sklearn.model_selection import ParameterGrid

from tigra.embeddings.embedding import Embedding
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.utils import n_nodes_to_ae_layers

from tigra.embeddings.sota.dynae_wrapper import DynAE
from tigra.embeddings.sota.dynrnn_wrapper import DynRNN
from tigra.embeddings.sota.dynaernn_wrapper import DynAERNN
from tigra.embeddings.sota.dyngem_wrapper import DynGEM

# Import new wrapper with penalty tensor support
from tigra.embeddings.ae.dynae_wrapper import DynAE as DynAE_v2


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
            # test creating an instance
            try:
                self.create_instance(i)
            except Exception as e:
                print("[tuner] failed to create instance")
                print("[tuner] configuration: " + str(self.confs[i]))
                print("[tuner] this configuration will not be accounted during tuning")
                print(e)
                continue

            prec_values, rec_values, f1_values = [], [], []
            for run_num in range(self.num_runs):
                current_instance = self.create_instance(i)
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
        print(self.params_desc)
        print(",".join([str(key) for key in self.confs[0].keys()]) + ",F1,PREC,REC")
        for i in range(len(self.confs)):
            s = ",".join([str(x) for x in self.confs[i].values()])
            eval = self.eval_res[i]
            s += "," + str(eval[1]) + "," + str(eval[2]) + "," + str(eval[3])
            print(s)

        print("Best conf: ", self.best, " [f1, prec, rec]")
        if self.best != None:
            print("Best conf params: ", self.confs[self.best[0]])

    def save_report(self, filename: str):
        reports = []
        for i, conf in enumerate(self.confs):
            report = {
                "config": conf,
                "scores": {
                    "f1": self.eval_res[i][1],
                    "prec": self.eval_res[i][2],
                    "rec": self.eval_res[i][3],
                },
            }
            reports.append(report)

        if self.best == None:
            best_report = None
        else:
            best_report = {
                "config": self.confs[self.best[0]],
                "scores": {
                    "f1": self.best[1],
                    "prec": self.best[2],
                    "rec": self.best[3],
                },
            }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "all": reports,
                    "best": best_report,
                },
                f,
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

    def __crop_n_units(self, n_units: typing.List[int]) -> typing.List[int]:
        if self.pick_n_layers_from_middle > len(n_units):
            print(
                "[tuner] DynamicEmbedding_Tuner instance field 'pick_n_layers_from_middle' larger than length of 'n_units'"
            )
            raise Exception(
                "field 'pick_n_layers_from_middle' larger than length of 'n_units'"
            )
        print("current config:", n_units)
        print("selecting n layers:", self.pick_n_layers_from_middle)
        padding_left = (len(n_units) - self.pick_n_layers_from_middle) / 2
        padding_left = math.floor(padding_left)
        print("left padding:", padding_left)
        n_units = n_units[padding_left:]
        n_units = n_units[: self.pick_n_layers_from_middle]
        print("selected config:", n_units)
        if len(n_units) == 0:
            print(
                "[tuner] DynamicEmbedding_Tuner config 'n_units' after removing padding resulted in no layers"
            )
            raise Exception(
                "config 'n_units' after removing padding resulted in no layers"
            )
        return n_units

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
            n_units = self.__crop_n_units(n_units)

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


class DynamicEmbedding2_Tuner(Tuner):
    """
    Tuner for DynamicEmbedding methods (DynGem, DynAE, DynRNN, DynAERNN)

    Variant without automatic parameter grid generation
    This is a workaround for easier tuning of Dynamic Embedding methods
    """

    Method = typing.Literal["DynAE", "DynRNN", "DynAERNN", "DynGEM"]

    method: Method
    configs: typing.List
    n_nodes: int
    has_cuda: bool

    def __init__(
        self,
        graph,
        data,
        params_desc,
        method: Method,
        configs: typing.List,
        has_cuda: bool,
        num_runs=1,
    ):
        super().__init__(graph, data, params_desc, num_runs)
        self.method = method
        self.configs = configs
        self.has_cuda = has_cuda

    def create_configurations(self) -> typing.List:
        return self.configs

    def create_instance(self, conf_index) -> Embedding:
        conf = self.confs[conf_index]

        if self.method == "DynAE":
            return DynAE(has_cuda=self.has_cuda, **conf)
        if self.method == "DynRNN":
            return DynRNN(has_cuda=self.has_cuda, **conf)
        if self.method == "DynAERNN":
            return DynAERNN(
                has_cuda=self.has_cuda,
                **conf,
            )
        if self.method == "DynGEM":
            return DynGEM(has_cuda=self.has_cuda, **conf)
        raise Exception(f"method '{self.method}' is incorrect")


class DynamicEmbedding3_Tuner(Tuner):
    """
    Tuner for new DynamicEmbedding methods with penalty tensor support

    This tuner uses the new wrapper implementation from tigra.embeddings.ae
    which supports penalty_type parameter ("beta" or "lid") and LID-based penalties.

    Unlike DynamicEmbedding2_Tuner, this only supports DynAE for now.
    """

    method: typing.Literal["DynAE"]
    configs: typing.List
    has_cuda: bool

    def __init__(
        self,
        graph,
        data,
        params_desc,
        configs: typing.List,
        has_cuda: bool,
        num_runs=1,
    ):
        """
        Parameters:
        -----------
        graph : Graph
            Referent graph used when reconstructing graph from embedding
        data : DynamicSparseMatrixGraph
            Dynamic graph data used to construct embedding
        params_desc : str
            Description of hyperparameters
        configs : List[dict]
            List of configuration dictionaries. Each config should contain:
            - dim: int - embedding dimension
            - look_back: int - number of historical snapshots to use
            - n_units: List[int] - architecture of autoencoder layers
            - beta: float - beta penalty weight (for penalty_type="beta")
            - nu1: float - L1 regularization weight
            - nu2: float - L2 regularization weight
            - batch_size: int - batch size for training
            - epochs: int - number of training epochs
            - learning_rate: float - learning rate
            - weight_decay: float - weight decay
            - penalty_type: str - "beta" or "lid" (optional, default="beta")
            - lid_k: int - k for LID computation (optional, default=10)
            - lid_cutoff: int - cutoff for LID shortest paths (optional, default=10)
            - use_lid_loss: bool - use GraphLIDLoss as additional loss (optional, default=False)
            - lambda_lid: float - weight for LID loss component (optional, default=0.1)
        has_cuda : bool
            Whether to use CUDA
        num_runs : int
            Number of runs per configuration (default=1)
        """
        super().__init__(graph, data, params_desc, num_runs)
        self.configs = configs
        self.has_cuda = has_cuda

    def create_configurations(self) -> typing.List:
        return self.configs

    def create_instance(self, conf_index) -> Embedding:
        conf = self.confs[conf_index]

        # Use new DynAE wrapper with penalty tensor support
        return DynAE_v2(has_cuda=self.has_cuda, **conf)
