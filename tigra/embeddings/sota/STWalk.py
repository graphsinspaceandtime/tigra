#####################
# STWalk1
# Learns trajectory representation of graph nodes by considering spatial and temporal neighbors together
#####################
import sys
from typing import List

sys.path.append("../")
sys.dont_write_bytecode = True


import random
from gensim.models import Word2Vec
import networkx as nx
import time

from tigra.embeddings.embedding import Embedding
from tigra.embeddings.embedding import SingletonEmbedding


class STWalk(Embedding):
    
    def __init__(self, number_restart, walk_length, dimension, time_step,
            time_window_size, workers, vocab_window_size):
        """
        This function generates representation for all nodes in space-time-graph of all nodes of graph at t=time_step
        however we will consider only representations of nodes present in graph at t = time_step
        """
        if time_window_size > time_step:
            sys.exit("ERROR: time_window_size(=" + str(time_window_size) + ") cannot be more than time_step(=" + str(
                time_step) + "):")

        self.number_restart = number_restart
        self.walk_length = walk_length
        self.dimension = dimension
        self.time_step = time_step
        self.time_window_size = time_window_size
        self.workers = workers
        self.vocab_window_size = vocab_window_size

    def createSpaceTimeGraph(self, G_list, time_window, start_node, time_step):
        """
        time step is necessary because we want representation only for last time step and
        we will create the space-time graph for [time_step, time_step-1,time_step-2,...,time_step-time_window]
        """
        G = G_list[-1]
        for time1 in range(1, time_window + 1):
            past_node = start_node.split("_")[0] + "_" + str(time_step - time1)
            if past_node not in G_list:
                continue
            else:
                G.add_edge(start_node, past_node)

                G_past = G_list[-time1 - 1]

                # considering first level neighbors
                past_neighbors = list(G_past.neighbors(past_node))
                temp = []

                # considering second level neighbors
                for elt in past_neighbors:
                    temp = temp + list(G_past.neighbors(elt))

                # merging list of level-1 and level-2 neighbors
                past_neighbors = past_neighbors + temp
                past_neighbors.append(past_node)

                # subgraph of G_past containing nodes from "past_neighbors" and edges between those nodes.
                past_subgraph = G_past.subgraph(past_neighbors)

                # merge current graph with past subgraphs
                G = nx.compose(G, past_subgraph)
                start_node = past_node
        return G


    def random_walk(self, SpaceTimegraph, path_length, rand=random.Random(0), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            removed = alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = SpaceTimegraph
        if start:
            path = [start]
        else:
            sys.exit("ERROR: Start node not mentioned for random_walk")

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                path.append(rand.choice(list(G[cur])))
            else:
                break
        return path


    def create_vocab(self, G_list, num_restart, path_length, nodes, time_step, rand=random.Random(0), time_window=1):
        walks = []

        nodes = list(nodes)

        # number of path is equal to number of restarts per node
        for cnt in range(num_restart):
            rand.shuffle(nodes)
            start = time.time()
            for node in list(nodes):
                G = self.createSpaceTimeGraph(G_list, time_window, str(node), time_step)
                walks.append(self.random_walk(SpaceTimegraph=G, path_length=path_length, rand=rand, start=node))
        #print("Vocabulary created")
        return walks


    def construct(self, data) -> List[SingletonEmbedding]:
        # get list of nodes
        graph_list = data.get_nx_graph_sequence()
        nodes = graph_list[-1].nodes()
        #print("Creating vocabulary...")
        seq = []

        for t in range(0, len(graph_list)):
            if (t + 1) % self.time_window_size == 0:
                seq.append(t)
        # time-step is decremented by 1, because, time steps are from 0 to time_step-1=total time_step length
        #print("Generating representation...")
        
        embedding = []
        for t in seq:
            walks = self.create_vocab(graph_list, num_restart=self.number_restart, path_length=self.walk_length, nodes=nodes,
                    rand=random.Random(0), time_step=t, time_window=self.time_window_size)
            model = Word2Vec(walks, window=self.vocab_window_size, min_count=0, workers=self.workers, vector_size=self.dimension)
            curr_emb = SingletonEmbedding()
            for node in nodes:
                curr_emb.add_embedding_vector(node, model.wv.get_vector(node))
            embedding.append(curr_emb)
        
        return embedding

