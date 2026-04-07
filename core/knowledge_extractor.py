import json
import time
from typing import List, Dict
from core.groq_client import get_client
from core.config import GROQ_MODEL


class KnowledgeExtractor:
    def __init__(self, model: str = GROQ_MODEL):
        self.client = get_client()
        self.model = model
        self.rate_limit_delay = 1

    def extract_entities(self, text: str, context: str = "general") -> List[Dict]:
        prompt = f"""Analyze this research paper text and extract key entities.

Context: This is a {context} research paper.
Text excerpt:
{text[:4000]}

Extract entities from these categories:
1. CONCEPT (theories, algorithms, models, frameworks)
2. METHOD (techniques, approaches, procedures)
3. TOOL (software, datasets, platforms)
4. METRIC (evaluation measures, benchmarks)
5. PROBLEM (challenges, issues addressed)

Return ONLY a JSON array:
[
  {{
    "name": "Exact entity name",
    "type": "CONCEPT|METHOD|TOOL|METRIC|PROBLEM",
    "description": "Brief description from context",
    "confidence": 0.95
  }}
]

Rules:
- Normalize names (e.g., "CNNs" → "Convolutional Neural Network")
- Confidence 0.0-1.0 based on clarity in text
- Return empty array [] if no entities found"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            time.sleep(self.rate_limit_delay)
            result = json.loads(response.choices[0].message.content)
            return result if isinstance(result, list) else result.get('entities', [])
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        entity_names = [e['name'] for e in entities]

        prompt = f"""Given this text and entities, identify relationships between them.

Text: {text[:4000]}

Entities: {entity_names}

Relationship types:
- IMPLEMENTS, USES, IMPROVES_UPON, EVALUATED_ON, ADDRESSES, COMPARES_TO

Return JSON array:
[
  {{
    "source": "Entity name",
    "target": "Entity name",
    "relation_type": "IMPLEMENTS|USES|IMPROVES_UPON|EVALUATED_ON|ADDRESSES|COMPARES_TO",
    "evidence": "Quote from text",
    "strength": 0.9
  }}
]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            time.sleep(self.rate_limit_delay)
            result = json.loads(response.choices[0].message.content)
            return result if isinstance(result, list) else result.get('relationships', [])
        except Exception as e:
            print(f"Relationship extraction error: {e}")
            return []