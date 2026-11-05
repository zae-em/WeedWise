"""A local, extractive botanical assistant with cited evidence and explicit abstention."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .knowledge import normalize

INTENTS={
    "chemical":{"herbicide","chemical","spray","als","accase","group","resistance"},
    "management":{"control","manage","management","remove","hoe","pull","cultural","mechanical","prevent","mulch"},
    "identification":{"identify","identification","look","features","characteristics","distinguish","leaves","description"},
    "taxonomy":{"family","scientific","latin","name","names","cycle","annual","crop","weed"},
}


class GroundedChatbot:
    def __init__(self,kb):
        self.kb=kb;self.claims=kb.evidence()
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),strip_accents="unicode",stop_words="english",sublinear_tf=True)
        documents=[" ".join(str(x) for x in [claim.get("species") or "general weeds",claim.get("scientific_name") or "",claim["category"],claim["text"]]) for claim in self.claims]
        self.matrix=self.vectorizer.fit_transform(documents)

    def answer(self,question,context_species=None):
        if not question.strip():return self._decline("Ask a question about a supported plant or weed-management topic.")
        if len(question)>2000:return self._decline("Please shorten the question to 2,000 characters.")
        q=normalize(question)
        if re.search(r"\b(dose|dosage|brand|brands|mixing|concentration)\b",q) or "application rate" in q or re.search(r"\b(ml|mg|litres|liters)\b",q):
            return self._decline("I do not provide pesticide brands, doses or mixing instructions. Consult a qualified local adviser and the approved product label. I can explain sourced herbicide groups and non-chemical management.")
        if any(term in q for term in ["ignore previous","ignore instructions","system prompt","invent a source","make up","without citations"]):
            return self._decline("I answer only from the curated, cited plant knowledge base.")
        found=self.kb.matching_species(question)
        if not found and context_species in self.kb.by_class:
            found=[self.kb.by_class[context_species]]
        words=set(q.split())
        if words & {"edible","eat","poisonous","toxicity","medicinal","medicine","weather","price","bitcoin"}:
            return self._decline("The curated records do not verify that topic. I cannot give food-safety, medicinal, weather or unrelated advice from these plant records.")
        if not found and not ("general" in words or "integrated" in words or "what is a weed" in q or "herbicide groups" in q or "mode of action" in q):
            return self._decline("Please name a supported species or explicitly ask for general integrated weed-management information. I have no verified species-specific answer for that question.")
        nonchemical=bool(re.search(r"\b(without|avoid|avoiding|no)\s+(chemical[s]?|herbicide[s]?|spray[s]?)\b",q) or "non chemical" in q)
        if nonchemical:categories={"cultural","mechanical","integrated","competition"}
        elif words & INTENTS["chemical"]:categories={"chemical","resistance"}
        elif words & INTENTS["management"]:categories={"cultural","mechanical","integrated","competition"}
        elif words & INTENTS["identification"]:categories={"description","identification","lookalikes"}
        elif words & {"compete","competition","harm","harmful","affected"}:categories={"competition","integrated"}
        elif words & INTENTS["taxonomy"]:categories={"taxonomy","life_cycle","classification"}
        else:categories=None
        scores=(self.matrix@self.vectorizer.transform([question]).T).toarray().ravel()
        allowed={record["class_label"] for record in found}
        candidates=[]
        for index,claim in enumerate(self.claims):
            if allowed and claim.get("species") not in allowed:continue
            if not allowed and claim.get("species") is not None:continue
            if categories and claim["category"] not in categories:continue
            if scores[index]>0.035 or allowed:candidates.append((float(scores[index]),index))
        candidates.sort(reverse=True)
        if len(found)>1:
            selected=[]
            for record in found:
                per_species=[self.claims[index] for _,index in candidates if self.claims[index].get("species")==record["class_label"]]
                selected.extend(per_species[:max(1,4//len(found))])
        else:selected=[self.claims[index] for _,index in candidates[:4]]
        if not selected:
            if allowed:
                return self._decline("The knowledge base has no verified information for that aspect of "+", ".join(sorted(allowed))+". I will not infer a treatment from another species.")
            return self._decline(f"I could not find supporting information in the curated knowledge base. Choose one of the {len(self.kb.records)} supported species or ask about general integrated weed management. I cannot identify an unsupported species from a text description.")
        source_ids=list(dict.fromkeys(source for claim in selected for source in claim["source_ids"]))
        references=self.kb.references(source_ids)
        sections=[]
        for claim in selected:
            refs=" ".join(f"[{source}]" for source in claim["source_ids"])
            label=(claim["species"]+": ") if len(found)>1 else ""
            sections.append(f"{label}{claim['text']} {refs}")
        prefix=(", ".join(record["common_name"] for record in found)+":\n\n") if found else "General source-backed guidance:\n\n"
        answer=prefix+"\n\n".join(sections)
        if categories and categories.intersection({"chemical","resistance","cultural","mechanical","integrated"}):
            answer+="\n\nThese sources describe their stated regions and contexts; suitability for your crop and location needs local verification."
        return {"supported":True,"answer":answer,"references":references,"evidence":selected,"mode":"local extractive retrieval; no external LLM"}

    @staticmethod
    def _decline(message):return {"supported":False,"answer":message,"references":[],"evidence":[],"mode":"local extractive retrieval"}
