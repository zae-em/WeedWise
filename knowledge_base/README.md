# Curated botanical knowledge base

species.json contains the 12 actual dataset classes, with claim-level source IDs. Tier A has more detailed
management coverage; Tier B contains shorter profiles and explicitly leaves unsupported treatments unavailable.
All sources were checked on 31 October 2026. They describe their own geographic and production contexts.
Pakistani suitability and species identity in local field photographs have not been independently verified.

The chatbot retrieves and displays existing claims; it uses no external LLM, secret API key, or invented text.
Draft species notes saved in SQLite remain awaiting expert verification and never become chatbot evidence.
Chemical information is limited to verified mode-of-action groups and resistance notes. No brands, doses,
mixing instructions or local product approvals are provided. See BOTANICAL_REVIEW.json for remaining review.

EPPO is used for names and taxonomy; university, public-sector and horticultural extension sources support
descriptions and management. The crop/weed label follows the dataset and is contextual rather than universal.
