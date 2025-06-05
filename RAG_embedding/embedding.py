import torch
import torch.nn.functional as F
import pandas as pd
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from tqdm import tqdm
import gc
import time

NEO4J_URI = "bolt://localhost:7687"  
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"

BATCH_SIZE = 100_000
MINI_BATCH_SIZE = 200

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def fetch_graph_batch(skip: int, limit: int) -> pd.DataFrame:
    query = f"""
    MATCH (s)-[r]->(o)
    WHERE s.embedding IS NULL OR o.embedding IS NULL
    RETURN elementId(s) AS src_id, elementId(o) AS dst_id,
           type(r) AS rel_type,
           labels(s)[0] AS src_type,
           labels(o)[0] AS dst_type
    SKIP {skip} LIMIT {limit}
    """
    with driver.session() as session:
        result = session.run(query)
        rows = result.data()
    return pd.DataFrame(rows)

def build_hetero_graph(df):
    data = HeteroData()

    node_types = set(df['src_type']).union(set(df['dst_type']))
    node_id_maps = {ntype: {} for ntype in node_types}
    node_idx_counters = {ntype: 0 for ntype in node_types}

    for row in df.itertuples():
        for ntype, nid in [(row.src_type, row.src_id), (row.dst_type, row.dst_id)]:
            if nid not in node_id_maps[ntype]:
                node_id_maps[ntype][nid] = node_idx_counters[ntype]
                node_idx_counters[ntype] += 1

    for ntype in node_types:
        num_nodes = len(node_id_maps[ntype])
        data[ntype].x = torch.randn(num_nodes, 128, dtype=torch.float32)

    edge_dict = {}
    for row in df.itertuples():
        src_idx = node_id_maps[row.src_type][row.src_id]
        dst_idx = node_id_maps[row.dst_type][row.dst_id]
        key = (row.src_type, row.rel_type, row.dst_type)
        edge_dict.setdefault(key, []).append((src_idx, dst_idx))

    for edge_type, connections in edge_dict.items():
        src_type, _, dst_type = edge_type
        if src_type not in data.node_types or dst_type not in data.node_types:
            continue
        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
        data[edge_type].edge_index = edge_index

    return data, node_id_maps

class HGTModel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=256, out_channels=300, heads=2):
        super().__init__()
        self.conv1 = HGTConv(128, hidden_channels, metadata, heads=heads)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=heads)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

def save_embeddings_to_neo4j(out_dict, node_id_maps):
    with driver.session() as session:
        for ntype, embeddings in out_dict.items():
            reverse_map = {v: k for k, v in node_id_maps[ntype].items()}
            embeddings_np = embeddings.cpu().detach().numpy()
            total = len(embeddings_np)
            bar_width = 40
            print(f"Saving embeddings for node type '{ntype}'...")
            for i in range(0, total, MINI_BATCH_SIZE):
                batch_data = [
                    {
                        "id": reverse_map[j],
                        "embedding": embeddings_np[j].tolist()
                    }
                    for j in range(i, min(i + MINI_BATCH_SIZE, total))
                ]
                query = f"""
                UNWIND $rows AS row
                MATCH (n:{ntype})
                WHERE elementId(n) = row.id AND n.embedding IS NULL
                SET n.embedding = row.embedding
                """
                try:
                    session.run(query, parameters={"rows": batch_data})
                    percent = int((i + MINI_BATCH_SIZE) / total * 100)
                    filled = int(bar_width * percent // 100)
                    bar = "-" * filled + " " * (bar_width - filled)
                    print(f"[{bar}] {percent}%", end='\r')
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[Neo4j ERROR] Batch {i//MINI_BATCH_SIZE + 1} failed: {e}")
                    continue
            print()

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skip = 0
    batch_count = 0

    while True:
        df = fetch_graph_batch(skip, BATCH_SIZE)
        if df.empty:
            print("No more data to process.")
            break

        print(f"\nProcessing batch {batch_count + 1}: SKIP {skip} LIMIT {BATCH_SIZE} ({len(df)} rows)")
        data, node_id_maps = build_hetero_graph(df)
        data.to(device)

        model = HGTModel(data.metadata()).to(device)

        print("Running model inference on GPU...")
        model.eval()
        with torch.no_grad():
            out_dict = model(data.x_dict, data.edge_index_dict)

        save_embeddings_to_neo4j(out_dict, node_id_maps)
        print(f" Embeddings created and saved for batch {batch_count + 1} (SKIP {skip})")

        del data, node_id_maps, out_dict, df
        gc.collect()
        torch.cuda.empty_cache()

        skip += BATCH_SIZE
        batch_count += 1

if __name__ == '__main__':
    main()
