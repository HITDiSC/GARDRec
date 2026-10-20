import os
from typing import Optional, Dict, Tuple

from neo4j import GraphDatabase

from gcn_detection import NEO4J_CONFIG as NEO4J_CONFIG_BASE


class Neo4jConnection:
    def __init__(self, uri, user, password, database="neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.connect()

    def connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def execute_query(self, query, parameters=None):
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return result.data()

    def find_common_nodes(self, seed_node_ids, min_connected_movies=2):
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        query = """
        WITH $movie_seed_ids AS seed_movie_ids
        MATCH (seed_movie:Movie)
        WHERE seed_movie.node_id IN seed_movie_ids
        MATCH (seed_movie)-[r]-(attr_node)
        WHERE type(r) IN ['BELONGS_TO_GENRE', 'PRODUCED_BY', 'DIRECTED', 'ACTED_IN', 'HAS_KEYWORD', 'WROTE']
          AND attr_node.node_id IS NOT NULL
        WITH attr_node, seed_movie,
            CASE type(r)
                WHEN 'BELONGS_TO_GENRE' THEN 'Genre'
                WHEN 'PRODUCED_BY' THEN 'Company'
                WHEN 'DIRECTED' THEN 'Director'
                WHEN 'ACTED_IN' THEN 'Actor'
                WHEN 'HAS_KEYWORD' THEN 'Keyword'
                WHEN 'WROTE' THEN 'Writer'
                ELSE 'Attribute'
            END AS node_type
        WITH attr_node, node_type,
            COUNT(DISTINCT seed_movie) AS connected_movie_count,
            COLLECT(DISTINCT seed_movie.node_id) AS connected_movie_ids
        WHERE connected_movie_count >= $min_connected_movies
        RETURN attr_node.node_id AS node_id,
            node_type,
            COALESCE(attr_node.name, attr_node.title, attr_node.node_id) AS node_name,
            connected_movie_count,
            connected_movie_ids,
            properties(attr_node) AS node_properties
        ORDER BY connected_movie_count DESC, node_type
        LIMIT 20
        """
        result = self.execute_query(
            query,
            parameters={
                "movie_seed_ids": seed_node_ids,
                "min_connected_movies": min_connected_movies,
            },
        )
        common_nodes = []
        for record in result:
            common_nodes.append({
                "node_id": record["node_id"],
                "node_type": record["node_type"],
                "node_name": record["node_name"],
                "connected_movie_count": record["connected_movie_count"],
                "connected_movie_ids": record["connected_movie_ids"],
                "node_properties": record["node_properties"],
            })
        return common_nodes[:20]


def get_neo4j_config(domain: str):
    config = dict(NEO4J_CONFIG_BASE)
    config["database"] = "kg-ml-1m" if domain == "ml1m" else "kg-ml-20m-else" if domain == "ml20m" else "kg-az-book"
    return config


def get_item_gcn_embedding(domain: str, limit: Optional[int] = None):
    config = get_neo4j_config(domain)
    uri = config.get("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user = config.get("username", config.get("user", os.getenv("NEO4J_USER", "neo4j")))
    password = config.get("password", os.getenv("NEO4J_PASSWORD", ""))
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)
    node_label = "Book" if domain == "amazonbook" else "Movie"
    id_field = "m.node_id" if domain == "amazonbook" else "m.original_id"
    embedding_dict = {}
    dim = 0
    with driver.session(database=config.get("database")) as session:
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
