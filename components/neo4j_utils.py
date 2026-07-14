import os
from typing import Optional, Dict, Tuple
from neo4j import GraphDatabase

from gcn_detection import NEO4J_CONFIG as NEO4J_CONFIG_BASE


def get_neo4j_config(domain: str):
    config = dict(NEO4J_CONFIG_BASE)
    config['database'] = "kg-ml-1m-en" if domain == "ml1m" else "kg-ml-20m-else" if domain == "ml20m" else "kg-az-book"
    return config


def get_item_gcn_embedding(domain: str, limit: Optional[int] = None):
    config = get_neo4j_config(domain)
    uri = config.get('uri', os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    user = config.get('username', config.get('user', os.getenv('NEO4J_USER', 'neo4j')))
    password = config.get('password', os.getenv('NEO4J_PASSWORD', ''))
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)
    node_label = "Book" if domain == "amazonbook" else "Movie"
    id_field = "m.node_id" if domain == "amazonbook" else "m.original_id"
    embedding_dict = {}
    dim = 0
    with driver.session(database=config.get('database')) as session:
        query = f"""
        MATCH (m:{node_label})
        WHERE m.embedding IS NOT NULL OR m.gcn_embedding IS NOT NULL OR m.hybrid_embedding IS NOT NULL
        RETURN {id_field} AS id, COALESCE(m.embedding, m.gcn_embedding, m.hybrid_embedding) AS emb
        """
        result = session.run(query)
        for record in result:
            emb = record["emb"]
            node_id = record["id"]
            if emb is not None and node_id is not None:
                embedding_dict[str(node_id)] = emb
                if dim == 0:
                    dim = len(emb)
    driver.close()
    return embedding_dict, dim
