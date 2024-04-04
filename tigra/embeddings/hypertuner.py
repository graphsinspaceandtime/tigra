"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module for tuning hyperparameters of dynamic graph embedding methods
"""

from abc import ABC, abstractmethod
from statistics import mean

from tigra.embeddings.embedding import Embedding
from tigra.core.greval import GraphReconstructionEval

from tigra.embeddings.sota.dynnode2vec import Dynnode2vec


class Tuner(ABC):
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
    def create_configurations(self):
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
                gre = GraphReconstructionEval(self.graph, last_embedding)
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

    
    def print_report(self):
        print(self.params_desc + ",F1,PREC,REC")
        for i in range(len(self.confs)):
            s = ','.join([str(x) for x in self.confs[i]])  
            eval = self.eval_res[i]
            s += "," + str(eval[1]) + "," + str(eval[2]) + "," + str(eval[3])
            print(s)

        print("Best conf: ", self.best)
        print("Best conf params: ", self.confs[self.best[0]])


EMBEDDING_DIMENSIONS = [10, 25, 50, 100, 200]
#EMBEDDING_DIMENSIONS = [10, 25]


class Dynnode2vec_Tuner(Tuner):
    def __init__(self, graph, data, p_values, q_values, num_runs=1, walk_length=10, n_walks_per_node=10, parallel_processes=4):
        super().__init__(graph, data, "DIM,P,Q", num_runs=num_runs)
        self.p_values = p_values
        self.q_values = q_values
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.parallel_processes = parallel_processes


    def create_configurations(self):
        confs = []
        for d in EMBEDDING_DIMENSIONS:
            for p in self.p_values:
                for q in self.q_values:
                    confs.append((d, p, q))

        return confs


    def create_instance(self, conf_index) -> Embedding:
        dim = self.confs[conf_index][0]
        p = self.confs[conf_index][1]
        q = self.confs[conf_index][2]
        return Dynnode2vec(dim, p=p, q=q, walk_length=self.walk_length, n_walks_per_node=self.n_walks_per_node, parallel_processes=self.parallel_processes)


