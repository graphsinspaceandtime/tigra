from graphviz import Source

core_code = """
digraph G {
    rankdir=TB;
    
    subgraph cluster_core {
        style=solid;
        color=black;
        bgcolor=gray;
        label="core";

        
        node [shape=ellipse, style=filled, fillcolor=white];
        "Graph";
        "Event";
        "EventSequence";
        "DynamicAgregateGraph";
        "GraphReconstructionEval";
        "DynamicSparseMatrixGraph";
        "DynamicGraph";
        "GraphComparator";
        "EvolDynamics";
        
        { rank=same; "Event";}
        { rank=same; "EventSequence"; "Graph"; }
        { rank=same; "DynamicAgregateGraph"; "DynamicGraph"; "GraphReconstructionEval"; "DynamicSparseMatrixGraph"; "GraphComparator";}
    }
    
    "SingletonEmbedding" [shape=ellipse, style=filled, fillcolor=white];

    "EventSequence" -> "Event";
    "DynamicGraph" -> "Graph";
    "DynamicGraph" -> "EventSequence";
    "DynamicAgregateGraph" -> "Graph";
    "DynamicAgregateGraph" -> "EventSequence";
    "DynamicSparseMatrixGraph" -> "Graph";
    "DynamicSparseMatrixGraph" -> "EventSequence";
    "GraphComparator" -> "Graph";
    "EvolDynamics" -> "DynamicGraph";
    "EvolDynamics" -> "GraphComparator";
    "GraphReconstructionEval" -> "GraphComparator";
    "GraphReconstructionEval" -> "SingletonEmbedding";
}
"""

# source = Source(core_code)
# source.render("repo-diagrams/core", format="png", cleanup=True)

emb_code = """
digraph G {
    rankdir=TB;
    
    subgraph cluster_core {
        style=solid;
        color=black;
        bgcolor=gray;
        label="embeddings";

        
        node [shape=ellipse, style=filled, fillcolor=white];
        "SingletonEmbedding";
        "Embedding";
        "STwalk";
        "Ctdne";
        "Dynnode2Vec";
        "Tuner";
        "N2VMethods_Tuner";
        "STWalk_Tuner";
        "DynamicEmbedding_Tuner";
        "Dynnode2vec_Tuner";
        "CTDNE_Tuner";
        
        { rank=same; "SingletonEmbedding";}
        { rank=same; "Embedding"; }
        { rank=same; "STwalk"; "Ctdne"; "Dynnode2Vec"; "Tuner"; }
        { rank=same; "N2VMethods_Tuner"; "STWalk_Tuner"; "DynamicEmbedding_Tuner"; }
        { rank=same; "Dynnode2vec_Tuner"; "CTDNE_Tuner" }

    }
    
    "Graph" [shape=ellipse, style=filled, fillcolor=white];
    "GraphReconstructionEval" [shape=ellipse, style=filled, fillcolor=white];
    
    { rank=same; "Graph";}

    "SingletonEmbedding" -> "Graph";
    "Embedding" -> "SingletonEmbedding";
    "STwalk" -> "Embedding" [label="extends"];
    "Ctdne" -> "Embedding" [label="extends"];
    "Dynnode2Vec" -> "Embedding" [label="extends"];
    "Tuner" -> "Embedding";
    "N2VMethods_Tuner" -> "Tuner" [label="extends"];
    "STWalk_Tuner" -> "Tuner" [label="extends"];
    "DynamicEmbedding_Tuner" -> "Tuner" [label="extends"];
    "Tuner" -> "GraphReconstructionEval";
    "Dynnode2vec_Tuner" -> "N2VMethods_Tuner" [label="extends"];
    "CTDNE_Tuner" -> "N2VMethods_Tuner" [label="extends"];
}
"""

# source = Source(emb_code)
# source.render("repo-diagrams/embeddings", format="png", cleanup=True)

eval_code = """
digraph G {
    rankdir=TB;
    
    subgraph cluster_core {
        style=solid;
        color=black;
        bgcolor=gray;
        label="evaluation";

        
        node [shape=ellipse, style=filled, fillcolor=white];
        "CumulativeStatistics";
        "CumulativeCorrelation";
        "CumulativeMWU";
        
        
        { rank=same; "CumulativeStatistics";}
        { rank=same; "CumulativeCorrelation"; "CumulativeMWU" }

    }
    
    "CumulativeCorrelation" -> "CumulativeStatistics" [label="extends"];
    "CumulativeMWU" -> "CumulativeStatistics" [label="extends"];
}
"""
# source = Source(eval_code)
# source.render("repo-diagrams/evaluation", format="png", cleanup=True)

gen_code = """
digraph G {
    rankdir=TB;
    
    subgraph cluster_core {
        style=solid;
        color=black;
        bgcolor=gray;
        label="generators";

        
        node [shape=ellipse, style=filled, fillcolor=white];
        "Generator";
        "DBAGenerator";
        "DBACGenerator";
        "DERGenerator";
        "DPPGenerator";
        "DSBMGenerator";
        "SBMGenerator";
        "SBTMGenerator";
        
        
        { rank=same; "Generator";}
    }
    
    "EventSequence" [shape=ellipse, style=filled, fillcolor=white];
    
    "Generator" -> "EventSequence";
    "DBAGenerator" -> "Generator" [label="extends"];
    "DBACGenerator" -> "Generator" [label="extends"];
    "DERGenerator" -> "Generator" [label="extends"];
    "DPPGenerator" -> "Generator" [label="extends"];
    "DSBMGenerator" -> "Generator" [label="extends"];
    "SBMGenerator" -> "Generator" [label="extends"];
    "SBTMGenerator" -> "Generator" [label="extends"];
}
"""

source = Source(gen_code)
source.render("repo-diagrams/generators", format="png", cleanup=True)
