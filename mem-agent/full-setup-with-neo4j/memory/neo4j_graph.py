import os

from dotenv import load_dotenv
from graph.models import GraphExtraction
from neo4j import GraphDatabase

load_dotenv()


class Neo4jGraph:
    def __init__(self):
        uri = os.environ["NEO_CONNECTION_URI"]
        username = os.environ["NEO_USERNAME"]
        password = os.environ["NEO_PASSWORD"]
        database = os.getenv("NEO_DATABASE", "neo4j")

        self.database = database

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        self.driver.verify_connectivity()

        print("Neo4j connection established.")



    def upsert_entity(
        self,
        user_id: str,
        name: str,
        entity_type: str,
    ):
        normalized_name = " ".join(name.strip().split()).lower()
    
        self.driver.execute_query(
            """
            MERGE (e:Entity {
                user_id: $user_id,
                normalized_name: $normalized_name
            })
            SET
                e.name = $name,
                e.type = $entity_type
            """,
            user_id=user_id,
            normalized_name=normalized_name,
            name=name.strip(),
            entity_type=entity_type,
            database_=self.database,
        )

        

    def upsert_relationship(
        self,
        user_id: str,
        source: str,
        relation: str,
        target: str,
    ):
        source_normalized = " ".join(source.strip().split()).lower()
        target_normalized = " ".join(target.strip().split()).lower()
    
        self.driver.execute_query(
            """
            MATCH (source:Entity {
                user_id: $user_id,
                normalized_name: $source_normalized
            })
    
            MATCH (target:Entity {
                user_id: $user_id,
                normalized_name: $target_normalized
            })
    
            MERGE (source)-[r:RELATES_TO {
                type: $relation
            }]->(target)
            """,
            user_id=user_id,
            source_normalized=source_normalized,
            target_normalized=target_normalized,
            relation=relation.strip().upper(),
            database_=self.database,
        )
    

    def upsert_graph(
        self,
        user_id: str,
        graph: GraphExtraction,
    ):
        for entity in graph.entities:
            self.upsert_entity(
                user_id=user_id,
                name=entity.name,
                entity_type=entity.type,
            )

        for relationship in graph.relationships:
            self.upsert_relationship(
                user_id=user_id,
                source=relationship.source,
                relation=relationship.relation,
                target=relationship.target,
            )


    def search(
        self,
        user_id: str,
        query: str,
    ):
        normalized_query = " ".join(query.strip().split()).lower()
    
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity {
                user_id: $user_id
            })
    
            WHERE e.normalized_name CONTAINS $query
    
            OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity {
                user_id: $user_id
            })
    
            RETURN
                e.name AS entity,
                e.type AS entity_type,
                r.type AS relation,
                other.name AS related_entity,
                other.type AS related_entity_type
    
            LIMIT 50
            """,
            user_id=user_id,
            query=normalized_query,
            database_=self.database,
        )
    
        return [record.data() for record in records]
        

    def get_entity_relationships(
        self,
        user_id: str,
        entity_name: str,
    ):
        normalized_name = " ".join(entity_name.strip().split()).lower()
    
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity {
                user_id: $user_id,
                normalized_name: $entity_name
            })
    
            OPTIONAL MATCH (e)-[out:RELATES_TO]->(target:Entity {
                user_id: $user_id
            })
    
            OPTIONAL MATCH (source:Entity {
                user_id: $user_id
            })-[inc:RELATES_TO]->(e)
    
            RETURN
                e.name AS entity,
                e.type AS entity_type,
    
                collect(DISTINCT {
                    direction: "outgoing",
                    relation: out.type,
                    related_entity: target.name,
                    related_entity_type: target.type
                }) AS outgoing,
    
                collect(DISTINCT {
                    direction: "incoming",
                    relation: inc.type,
                    related_entity: source.name,
                    related_entity_type: source.type
                }) AS incoming
            """,
            user_id=user_id,
            entity_name=normalized_name,
            database_=self.database,
        )
    
        return [record.data() for record in records]


    def traverse(
        self,
        user_id: str,
        entity_name: str,
        max_hops: int = 2,
    ):
        normalized_name = " ".join(entity_name.strip().split()).lower()
    
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1")
    
        max_hops = min(max_hops, 5)
    
        records, _, _ = self.driver.execute_query(
            f"""
            MATCH (start:Entity {{
                user_id: $user_id,
                normalized_name: $entity_name
            }})
    
            MATCH path =
                (start)-[:RELATES_TO*1..{max_hops}]-(related:Entity {{
                    user_id: $user_id
                }})
    
            UNWIND relationships(path) AS rel
    
            RETURN DISTINCT
                start.name AS start_entity,
                start.type AS start_entity_type,
                startNode(rel).name AS source,
                rel.type AS relation,
                endNode(rel).name AS target,
                endNode(rel).type AS target_type
    
            LIMIT 200
            """,
            user_id=user_id,
            entity_name=normalized_name,
            database_=self.database,
        )
    
        return [record.data() for record in records]


    def get_user_graph(
        self,
        user_id: str,
    ):
        records, _, _ = self.driver.execute_query(
            """
            MATCH (source:Entity {
                user_id: $user_id
            })
    
            OPTIONAL MATCH (source)-[r:RELATES_TO]->(target:Entity {
                user_id: $user_id
            })
    
            RETURN
                source.name AS source,
                source.type AS source_type,
                r.type AS relation,
                target.name AS target,
                target.type AS target_type
    
            LIMIT 200
            """,
            user_id=user_id,
            database_=self.database,
        )
    
        return [
            record.data()
            for record in records
        ]

    
    def close(self):
        self.driver.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    graph = Neo4jGraph()

    graph.upsert_entity(
        user_id="lakshya",
        name="Lakshya",
        entity_type="Person",
    )

    graph.upsert_entity(
        user_id="lakshya",
        name="AI Coding Agent",
        entity_type="Project",
    )

    graph.upsert_relationship(
        user_id="lakshya",
        source="Lakshya",
        relation="BUILDING",
        target="AI Coding Agent",
    )

    print("\nGraph search:")

    results = graph.search(
        user_id="lakshya",
        query="Lakshya",
    )

    for result in results:
        print(result)

    graph.close()
