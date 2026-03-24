import sys
import os


sys.path.append(os.path.join(".."))
sys.dont_write_bytecode = True


from tigra.embeddings.sota.factory import get_rl_walk_factory
try:
    from tigra.data.bank import Databank
except ImportError:
    try:
        from data.bank import Databank
    except ImportError:
        from tigra.data.loaders import Databank


CONF_PATH = "../rl_walk1.conf" 

print(f"Loading Factory with config: {CONF_PATH}")
rl_factory = get_rl_walk_factory(CONF_PATH)

datasets = [
    "radoslaw-email",
    "ia-hospital-ward-proximity-attr",
    "ia-contacts_hypertext2009",
    "ia-enron-employees",
    "ia-primary-school-proximity-attr",
    "fb-forum",
    "email-Eu-core-temporal",
    "CollegeMsg",
    "fb-messages",
]

for dataset in datasets:
    print(f"PROCESSING: {dataset}")
    
    try:
        rl_model, dg = rl_factory.get_embedding(dataset)
        
        if rl_model is not None:
            rl_model.save_embedding(dataset)
            print(f"SUCCESS: Saved embeddings for {dataset}")
        else:
            print(f"SKIP: No config found for {dataset}")
            
    except Exception as e:
        print(f"ERROR processing {dataset}: {e}")