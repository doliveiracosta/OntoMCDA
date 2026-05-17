"""OntoMCDA semantic recommendation package."""

from .ontology import load_profiles
from .recommender import RecommendationResult, recommend_methods
from .text_inference import infer_premises

__all__ = ["RecommendationResult", "infer_premises", "load_profiles", "recommend_methods"]
