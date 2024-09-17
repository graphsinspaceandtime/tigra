"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module for tuning hyperparameters of dynamic graph embedding methods
"""

from abc import ABC, abstractmethod
import typing

from statistics import mean

from tigra.embeddings.embedding import Embedding
from tigra.core.greval import GraphReconstructionEval

from tigra.embeddings.sota.dynae_wrapper import DynAE

# from tigra.embeddings.sota.dynnode2vec import Dynnode2vec
# from tigra.embeddings.sota.ctdne_wrapper import Ctdne
# from tigra.embeddings.sota.stwalk import STWalk


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

    def tune(self):
        self.best = None
        self.eval_res = []
        self.confs = self.create_configurations()

        for i in range(len(self.confs)):
            prec_values, rec_values, f1_values = [], [], []
            for _ in range(self.num_runs):
                current_instance = self.create_instance(i)
                current_instance.make_embedding(self.data)
                last_embedding = current_instance.embeddings[-1]
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

            if i % 10 == 0:
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

    # def create_instance(self, conf_index) -> Embedding:
    #     dim = self.confs[conf_index][0]
    #     p = self.confs[conf_index][1]
    #     q = self.confs[conf_index][2]
    #     wl = self.confs[conf_index][3]
    #     nw = self.confs[conf_index][4]
    #     return Dynnode2vec(
    #         dim,
    #         p=p,
    #         q=q,
    #         walk_length=wl,
    #         n_walks_per_node=nw,
    #         parallel_processes=self.parallel_processes,
    #     )


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

    # def create_instance(self, conf_index) -> Embedding:
    #     dim = self.confs[conf_index][0]
    #     p = self.confs[conf_index][1]
    #     q = self.confs[conf_index][2]
    #     wl = self.confs[conf_index][3]
    #     nw = self.confs[conf_index][4]
    #     return Ctdne(
    #         dim,
    #         p,
    #         q,
    #         walk_length=wl,
    #         num_walks=nw,
    #         workers=self.parallel_processes,
    #     )


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

    # def create_instance(self, conf_index) -> Embedding:
    #     dim = self.confs[conf_index][0]
    #     tws = self.confs[conf_index][1]
    #     ts = len(self.data.get_nx_graph_sequence())
    #     return STWalk(
    #         number_restart=self.n_walks_per_node,
    #         walk_length=self.walk_length,
    #         dimension=dim,
    #         time_step=ts,
    #         time_window_size=tws,
    #         workers=self.parallel_processes,
    #         vocab_window_size=5,
    #     )


class AE_Tuner(Tuner):
    """
    Tuner for Autoencoders
    """

    def __init__(
        self,
        graph,
        data,
        n_units,
        look_back,
        beta,
        nu1,
        nu2,
        batch_size,
        learning_rate,
        weight_decay,
        epochs,
    ):
        super().__init__(graph, data, "N_UNITS, LOOK_BACK, BETA, NU1, NU2")
        self.n_units = n_units
        self.look_back = look_back
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

    def create_instance(self, conf_index) -> Embedding:
        dimension = self.confs[conf_index][0]
        n_units = self.confs[conf_index][1]
        look_back = self.confs[conf_index][2]
        beta = self.confs[conf_index][3]
        nu1 = self.confs[conf_index][4]
        nu2 = self.confs[conf_index][5]

        return DynAE(
            dim=dimension,
            n_units=n_units,
            look_back=look_back,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            bias=True,
            shuffle=True,
            has_cuda=False,
        )

    def create_configurations(self):
        confs = []

        for d in EMBEDDING_DIMENSIONS:
            for n_u in self.n_units:
                for l_b in self.look_back:
                    for b in self.beta:
                        for n1 in self.nu1:
                            for n2 in self.nu2:
                                confs.append((d, n_u, l_b, b, n1, n2))

        return confs
