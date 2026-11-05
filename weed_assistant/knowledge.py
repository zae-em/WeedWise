"""Curated claim-level citations. Unreviewed user drafts are never chatbot evidence."""
import json
import re
from pathlib import Path
from .config import KNOWLEDGE_PATH


def normalize(text):return " ".join(re.sub(r"[^a-z0-9]+"," ",text.lower().replace("’","").replace("'","")).split())


class KnowledgeBase:
    def __init__(self,path=KNOWLEDGE_PATH):
        self.path=Path(path)
        payload=json.loads(self.path.read_text(encoding="utf-8"))
        self.records=payload["species"]
        self.sources={row["id"]:row for row in payload["sources"]}
        self.general=payload.get("general",[])
        self.by_class={row["class_label"]:row for row in self.records}
        self.validate()

    def validate(self):
        if len(self.by_class)!=len(self.records):raise ValueError("Duplicate knowledge-base class")
        for row in self.records:
            if row["tier"] not in {"A","B"}:raise ValueError("Invalid knowledge tier")
            for claim in row["claims"]:
                if not claim.get("source_ids") or any(source not in self.sources for source in claim["source_ids"]):
                    raise ValueError(f"Unsourced claim in {row['class_label']}")
        for claim in self.general:
            if not claim.get("source_ids") or any(source not in self.sources for source in claim["source_ids"]):raise ValueError("Unsourced general claim")

    def get(self,label):return self.by_class.get(label)

    def matching_species(self,question):
        question=" "+normalize(question)+" "
        hits=[]
        for record in self.records:
            aliases=[record["class_label"],record["common_name"],record["scientific_name"]]+record.get("aliases",[])
            if any(" "+normalize(alias)+" " in question for alias in aliases if alias):hits.append(record)
        return hits

    def evidence(self):
        result=[]
        for record in self.records:
            for claim in record["claims"]:
                result.append({**claim,"species":record["class_label"],"scientific_name":record["scientific_name"]})
        result.extend({**claim,"species":None,"scientific_name":None} for claim in self.general)
        return result

    def references(self,source_ids):return [self.sources[source] for source in dict.fromkeys(source_ids)]
