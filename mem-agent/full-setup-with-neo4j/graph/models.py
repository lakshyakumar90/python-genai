from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str
    type: str


class Relationship(BaseModel):
    source: str
    relation: str
    target: str


class GraphExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)