import re
from typing import List, Dict, Optional
from agents.state import GraphPath
from core.vector_db import get_collection
from core.graph_db import get_driver


def tool_vector_search(query: str, top_k: int = 5, paper_filter: List[str] = None) -> List[Dict]:
    """Retrieve relevant text chunks from ChromaDB"""
    try:
        collection = get_collection()
        where_filter = None
        if paper_filter:
            where_filter = {"paper_id": {"$in": paper_filter}}

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        formatted = []
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            formatted.append({
                'text': results['documents'][0][i],
                'paper_id': meta.get('paper_id', 'unknown'),
                'section': meta.get('section', 'unknown'),
                'chunk_index': meta.get('chunk_index', 0),
                'similarity_score': 1 - results['distances'][0][i]
            })
        return formatted

    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def tool_graph_lookup(entity_name: str, relation_type: str = None) -> List[Dict]:
    """Find direct neighbors of an entity in Neo4j (1-hop)"""
    try:
        driver = get_driver()
        with driver.session() as session:
            if relation_type:
                query = """
                MATCH (e:Entity)-[r:RELATES_TO {relation_type: $rel_type}]-(neighbor:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN e.name as source, neighbor.name as target,
                       r.relation_type as relation, r.evidence as evidence, r.strength as strength
                LIMIT 15
                """
                result = session.run(query, {'entity_name': entity_name, 'rel_type': relation_type})
            else:
                query = """
                MATCH (e:Entity)-[r:RELATES_TO]-(neighbor:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN e.name as source, neighbor.name as target,
                       r.relation_type as relation, r.evidence as evidence, r.strength as strength
                LIMIT 15
                """
                result = session.run(query, {'entity_name': entity_name})
            return [dict(record) for record in result]

    except Exception as e:
        print(f"Graph lookup error: {e}")
        return []


def tool_multihop_paths(start_entity: str, end_entity: str = None, max_hops: int = 3) -> List[GraphPath]:
    """Find paths through knowledge graph (2-3 hops)"""
    paths = []
    try:
        driver = get_driver()
        with driver.session() as session:
            if end_entity:
                query = f"""
                MATCH path = (start:Entity)-[r:RELATES_TO*1..{max_hops}]-(end:Entity)
                WHERE toLower(start.name) CONTAINS toLower($start_name)
                  AND toLower(end.name) CONTAINS toLower($end_name)
                RETURN [n IN nodes(path) | n.name] as node_names,
                       [rel IN relationships(path) | rel.relation_type] as rel_types,
                       [rel IN relationships(path) | rel.evidence] as evidence_list,
                       length(path) as path_length
                ORDER BY path_length LIMIT 5
                """
                result = session.run(query, {'start_name': start_entity, 'end_name': end_entity})
            else:
                query = f"""
                MATCH path = (start:Entity)-[r:RELATES_TO*2..{max_hops}]-(end:Entity)
                WHERE toLower(start.name) CONTAINS toLower($start_name)
                  AND start <> end
                RETURN [n IN nodes(path) | n.name] as node_names,
                       [rel IN relationships(path) | rel.relation_type] as rel_types,
                       [rel IN relationships(path) | rel.evidence] as evidence_list,
                       length(path) as path_length
                ORDER BY path_length LIMIT 10
                """
                result = session.run(query, {'start_name': start_entity})

            for record in result:
                paths.append(GraphPath(
                    nodes=record['node_names'],
                    relationships=record['rel_types'],
                    evidence=record['evidence_list'],
                    path_length=record['path_length']
                ))
        return paths

    except Exception as e:
        print(f"Multi-hop error: {e}")
        return []