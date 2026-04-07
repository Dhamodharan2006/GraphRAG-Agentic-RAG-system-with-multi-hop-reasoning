from neo4j import GraphDatabase
from core.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver

def graph_lookup(entity_name: str, relation_type: str = None):
    driver = get_driver()
    with driver.session() as session:
        if relation_type:
            query = """
            MATCH (e:Entity {name: $name})-[r]->(neighbor)
            WHERE type(r) = $rel
            RETURN neighbor.name as name, type(r) as relation, labels(neighbor) as labels
            LIMIT 20
            """
            result = session.run(query, name=entity_name, rel=relation_type)
        else:
            query = """
            MATCH (e:Entity {name: $name})-[r]->(neighbor)
            RETURN neighbor.name as name, type(r) as relation, labels(neighbor) as labels
            LIMIT 20
            """
            result = session.run(query, name=entity_name)
        return [dict(r) for r in result]

def multihop_paths(start: str, end: str = None, max_hops: int = 3):
    driver = get_driver()
    with driver.session() as session:
        query = f"""
        MATCH path = (s:Entity {{name: $start}})-[*1..{max_hops}]->(t:Entity)
        WHERE t.name <> $start
        RETURN [n in nodes(path) | n.name] as nodes,
               [r in relationships(path) | type(r)] as rels
        LIMIT 10
        """
        result = session.run(query, start=start)
        return [dict(r) for r in result]