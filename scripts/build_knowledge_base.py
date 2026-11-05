"""Curate concise source-attributed records for the actual 12 dataset labels."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

SOURCES=[
 ("dataset","Aarhus University: Plant Seedlings Dataset","https://vision.eng.au.dk/plant-seedlings-dataset/","Denmark / dataset provenance"),
 ("black_bio","AHDB: Basic black-grass biology","https://ahdb.org.uk/knowledge-library/arable-weeds-basic-black-grass-biology","United Kingdom"),
 ("black_iwm","AHDB: Integrated management of black-grass","https://ahdb.org.uk/knowledge-library/integrated-management-of-black-grass","United Kingdom / arable crops"),
 ("charlock_bio","AHDB: Charlock biology","https://ahdb.org.uk/knowledge-library/distribution-and-biology-of-charlock-in-the-uk","United Kingdom / arable crops"),
 ("cleavers_rhs","RHS: Cleavers","https://www.rhs.org.uk/weeds/cleavers","United Kingdom / gardens"),
 ("cleavers_dk","Crop Protection Online: Cleavers biology","https://plantevaernonline.dlbr.dk/cp/graphics/Name.asp?DatasourceID=1&Language=en&NameID=145&TaskID=1&id=djf","Denmark / crops"),
 ("chickweed_bio","Iowa State University: Common chickweed","https://crops.extension.iastate.edu/encyclopedia/common-chickweed","United States"),
 ("chickweed_rhs","RHS: Chickweed","https://www.rhs.org.uk/weeds/chickweed","United Kingdom / gardens"),
 ("fat_hen_awri","Australian Wine Research Institute: Fat hen","https://www.awri.com.au/industry_support/viticulture/weed-management/weeds/fat-hen/","Australia / viticulture"),
 ("silky_bio","AHDB: Loose silky bent biology","https://ahdb.org.uk/knowledge-library/distribution-and-biology-of-loose-silky-bent-in-the-uk","United Kingdom / cereals"),
 ("mayweed_bio","AHDB: Scentless mayweed biology","https://ahdb.org.uk/knowledge-library/distribution-and-biology-of-scentless-mayweed-in-the-uk","United Kingdom / arable crops"),
 ("mayweed_nwcb","Washington State Noxious Weed Control Board: Scentless Mayweed","https://www.nwcb.wa.gov/weeds/scentless-mayweed","Washington State, USA"),
 ("purse_bio","Iowa State University: Shepherds purse","https://crops.extension.iastate.edu/encyclopedia/shepherds-purse","United States"),
 ("purse_wsu","Washington State University Hortsense: Shepherd's-purse","https://hortsense.cahnrs.wsu.edu/fact-sheet/weeds-shepherds-purse-capsella-bursa-pastoris/","Washington State, USA"),
 ("geranium_bio","Native Plant Trust Go Botany: Geranium pusillum","https://gobotany.nativeplanttrust.org/species/geranium/pusillum/","New England, USA / morphology"),
 ("wheat_ncsu","NC State Extension: Triticum aestivum","https://plants.ces.ncsu.edu/plants/triticum-aestivum/","United States / horticulture"),
 ("maize_ncsu","NC State Extension: Zea mays","https://plants.ces.ncsu.edu/plants/zea-mays/","United States / horticulture"),
 ("uc_iwm","University of California IPM: Weed Management in Landscapes","https://ipm.ucanr.edu/home-and-landscape/weed-management-in-landscapes/","California, USA / landscape beds"),
 ("hrac","HRAC Global: Herbicide classification lookup","https://hracglobal.com/tools/classification-lookup?sort=hrac","Global / mode-of-action terminology"),
]

BASE=[
 ("Black-grass","Black-grass","Alopecurus myosuroides","Poaceae","weed","grass","Annual","A","ALOMY",["blackgrass","black grass","slender foxtail"]),
 ("Charlock","Charlock","Sinapis arvensis","Brassicaceae","weed","broadleaf","Annual","B","SINAR",["wild mustard"]),
 ("Cleavers","Cleavers","Galium aparine","Rubiaceae","weed","broadleaf","Annual","A","GALAP",["catchweed bedstraw","goosegrass"]),
 ("Common Chickweed","Common chickweed","Stellaria media","Caryophyllaceae","weed","broadleaf","Winter or summer annual","A","STEME",["chickweed","winterweed"]),
 ("Common wheat","Common wheat","Triticum aestivum","Poaceae","crop","grass","Annual","B","TRZAX",["wheat","bread wheat","winter wheat"]),
 ("Fat Hen","Fat hen","Chenopodium album","Amaranthaceae","weed","broadleaf","Annual","A","CHEAL",["fat-hen","lambsquarters","lamb's quarters","white goosefoot"]),
 ("Loose Silky-bent","Loose silky-bent","Apera spica-venti","Poaceae","weed","grass","Annual","B","APESV",["loose silky bent","silky bent","wind grass"]),
 ("Maize","Maize","Zea mays","Poaceae","crop","grass","Annual","B","ZEAMX",["corn"]),
 ("Scentless Mayweed","Scentless mayweed","Tripleurospermum inodorum","Asteraceae","weed","broadleaf","Annual to short-lived perennial; varies by region","A","MATIN",["scentless chamomile","Matricaria perforata"]),
 ("Shepherds Purse","Shepherd's purse","Capsella bursa-pastoris","Brassicaceae","weed","broadleaf","Winter annual; regional variation reported","B","CAPBP",["shepherds purse","shepherd's-purse"]),
 ("Small-flowered Cranesbill","Small-flowered cranesbill","Geranium pusillum","Geraniaceae","weed","broadleaf","Not independently verified in this record","B","GERPU",["small flowered cranesbill","small-flower geranium"]),
 ("Sugar beet","Sugar beet","Beta vulgaris subsp. vulgaris var. altissima","Amaranthaceae","crop","broadleaf","Not independently verified for this crop record","B","BEAVA",["sugarbeet","Beta vulgaris"]),
]


def build():
    sources=[{"id":identifier,"title":title,"url":url,"context":context,"accessed":"2026-10-31"} for identifier,title,url,context in SOURCES]
    records=[]
    def claim(record,category,text,*source_ids):
        record["claims"].append({"category":category,"text":text,"source_ids":list(source_ids),"verification":"source_checked; not independently field-verified"})
    for label,common,scientific,family,role,plant_type,life_cycle,tier,eppo,aliases in BASE:
        source_id="eppo_"+eppo
        sources.append({"id":source_id,"title":"EPPO Global Database: "+scientific,"url":"https://gd.eppo.int/taxon/"+eppo,"context":"Taxonomy and names; not a local management approval","accessed":"2026-10-31"})
        record={"id":eppo.lower(),"class_label":label,"common_name":common,"scientific_name":scientific,"family":family,"role":role,"plant_type":plant_type,
                "life_cycle":life_cycle,"tier":tier,"aliases":aliases,"taxonomy_source_ids":[source_id,"dataset"],"pakistan_relevance":"Not verified for this record",
                "expert_review_status":"Not independently reviewed by a local botanist/agronomist","claims":[]}
        claim(record,"taxonomy",f"{common} is recorded here as {scientific}, in the family {family}.",source_id)
        claim(record,"classification",f"The dataset treats {common} as a {role} class. This project label does not decide whether a plant is unwanted in your field.","dataset","cleavers_rhs")
        records.append(record)
    r={record["class_label"]:record for record in records}
    claim(r["Black-grass"],"description","Black-grass is an annual tufted grass. Young leaves are fine and smooth; older flowering plants bear narrow, dark-purplish heads.","black_bio")
    claim(r["Black-grass"],"lookalikes","Young black-grass can resemble loose silky-bent; a purple or reddish leaf sheath alone is insufficient to distinguish them.","black_bio","silky_bio")
    claim(r["Black-grass"],"cultural","AHDB's UK arable guidance combines rotation changes, suitable spring cropping, competitive crops and delayed autumn sowing to reduce black-grass pressure. Timing must be adapted locally.","black_iwm")
    claim(r["Black-grass"],"mechanical","AHDB describes rotational inversion tillage and hand-roguing low-density patches as parts of black-grass control. Prevent surviving plants returning seed.","black_iwm")
    claim(r["Black-grass"],"integrated","Use several complementary measures and reduce seed movement on machinery or infested material; non-chemical measures also help protect herbicide efficacy.","black_iwm")
    claim(r["Charlock"],"description","Charlock is a hairy annual broadleaf with irregularly toothed leaves and pale-yellow flowers with four petals.","charlock_bio")
    claim(r["Charlock"],"competition","AHDB reports competition with winter wheat and winter oilseed rape in the UK. Control is particularly difficult in brassica crops; this is not a Pakistani treatment recommendation.","charlock_bio")
    claim(r["Cleavers"],"description","Cleavers has scrambling stems, whorled leaves and hooked bristles that cling to clothing and other plants. Tiny white flowers later form bristly fruits.","cleavers_rhs")
    claim(r["Cleavers"],"life_cycle","Cleavers is an annual; the Danish crop-protection guide describes both summer- and winter-annual growth.","cleavers_dk")
    claim(r["Cleavers"],"mechanical","In garden beds, remove rooted cleavers plants or hoe young seedlings before flowering and seed production. Minimize unnecessary soil disturbance.","cleavers_rhs")
    claim(r["Cleavers"],"cultural","RHS garden guidance includes mulching bare soil to suppress new cleavers seedlings. Recheck boundaries and young plantings for new growth.","cleavers_rhs")
    claim(r["Cleavers"],"competition","The Danish crop guide reports that cleavers can contribute to lodging and crop losses; the significance depends on crop and infestation.","cleavers_dk")
    claim(r["Common Chickweed"],"description","Common chickweed forms low, fragile stems with opposite, pointed oval leaves. Its white petals are deeply notched; mouseear chickweed has more densely hairy leaves.","chickweed_bio")
    claim(r["Common Chickweed"],"life_cycle","Iowa State describes common chickweed as a winter or summer annual.","chickweed_bio")
    claim(r["Common Chickweed"],"mechanical","Pull shallow-rooted plants or hoe seedlings on a dry day before they set seed. Keep flowering material out of ordinary home compost where seed may survive.","chickweed_rhs")
    claim(r["Common Chickweed"],"cultural","For garden beds, mulch or suitable ground cover can suppress new chickweed seedlings. Avoid unnecessary soil disturbance that brings buried seed to the surface.","chickweed_rhs")
    claim(r["Common wheat"],"description","Common wheat is an annual cereal grass cultivated for grain, forage or cover cropping; its leaves are linear.","wheat_ncsu")
    claim(r["Common wheat"],"cultural","NC State describes wheat as a cover crop that can suppress weeds. This does not authorize removing a desired wheat crop or selecting a herbicide for it.","wheat_ncsu")
    claim(r["Fat Hen"],"description","Fat hen is an erect annual herb with elliptic to diamond-shaped, toothed leaves and a pale mealy lower surface. Flowers occur in dense small green clusters.","fat_hen_awri")
    claim(r["Fat Hen"],"competition","AWRI describes competition for water and nutrients in its viticulture context; regional crop relevance requires separate verification.","fat_hen_awri")
    claim(r["Fat Hen"],"mechanical","AWRI describes mowing before flowering to reduce seed set and hand-pulling before seeding. This information comes from Australian vineyard management.","fat_hen_awri")
    claim(r["Fat Hen"],"integrated","Monitor new growth after rain or irrigation and prevent seed production; the source describes rapid growth in moist conditions and long-lived buried seed.","fat_hen_awri")
    claim(r["Loose Silky-bent"],"description","Loose silky-bent is an annual grass with smooth leaf sheaths and an open, feathery flowerhead. It can resemble young black-grass.","silky_bio")
    claim(r["Loose Silky-bent"],"cultural","AHDB reports that grass breaks or spring cropping can reduce loose silky-bent populations in UK cereal rotations.","silky_bio")
    claim(r["Loose Silky-bent"],"mechanical","In reduced-tillage settings, AHDB describes allowing emergence before cultivating or harrowing seedlings when soil is dry.","silky_bio")
    claim(r["Loose Silky-bent"],"resistance","Resistance to ALS inhibitors is reported in parts of mainland Europe. ALS inhibitors are HRAC Group 2; this information does not establish local resistance or efficacy.","silky_bio","hrac")
    claim(r["Maize"],"description","Maize, also called corn, is an annual grass crop in Poaceae. It is wind-pollinated and cultivated for its grain and other crop uses.","maize_ncsu")
    claim(r["Maize"],"cultural","NC State's garden guidance favors mulch around established corn rather than cultivation that may injure its roots. Do not interpret a maize prediction as a weed-removal instruction.","maize_ncsu")
    claim(r["Scentless Mayweed"],"description","Scentless mayweed has finely divided, thread-like leaves and white daisy-like flowerheads with yellow centers. Non-flowering mayweeds can be difficult to distinguish.","mayweed_bio","mayweed_nwcb")
    claim(r["Scentless Mayweed"],"life_cycle","Regional guides describe annual to short-lived perennial behavior. A growth-stage photograph alone does not establish the full life cycle.","mayweed_nwcb")
    claim(r["Scentless Mayweed"],"mechanical","Washington State guidance supports hand-pulling small infestations. Mowing can slow flowering, but plants may flower again below the cut height.","mayweed_nwcb")
    claim(r["Scentless Mayweed"],"cultural","Competitive vegetation can suppress scentless mayweed, and cleaning equipment helps limit movement into new areas.","mayweed_nwcb")
    claim(r["Scentless Mayweed"],"resistance","AHDB reports UK populations resistant to ALS inhibitors. These are HRAC Group 2. Resistance testing and local professional advice are needed before a control programme is chosen.","mayweed_bio","hrac")
    claim(r["Shepherds Purse"],"description","Shepherd's purse forms a basal leaf rosette; flowering stems bear distinctive triangular or heart-shaped seed pods. Older rosette leaves can be deeply lobed.","purse_bio")
    claim(r["Shepherds Purse"],"life_cycle","Iowa State describes a winter annual; Washington State notes annual, winter-annual and occasional biennial behavior.","purse_bio","purse_wsu")
    claim(r["Small-flowered Cranesbill"],"description","Small-flowered cranesbill has opposite, hairy leaves that are rounded to kidney-shaped, with palm-like lobes. Flowers have five separate petals.","geranium_bio")
    claim(r["Sugar beet"],"description","Sugar beet is a cultivated beet variety group grown as a field crop for sugar extraction, in Amaranthaceae. It is one of the three crop labels in this dataset.","eppo_BEAVA","dataset")
    general=[
      {"category":"integrated","text":"Integrated weed management combines identification, site assessment, competitive planting, soil-cover methods and timely weed removal. Tactics depend on the planting system; landscape guidance is not a crop-specific field prescription.","source_ids":["uc_iwm"]},
      {"category":"cultural","text":"In landscape beds, competitive plants and suitable mulches reduce exposed soil and suppress new weeds. The UC source recommends maintaining coverage and monitoring new emergence.","source_ids":["uc_iwm"]},
      {"category":"mechanical","text":"General landscape guidance includes removing young weeds before seed production and shallow hoeing for annual weeds, while protecting desirable plants and limiting unnecessary disturbance.","source_ids":["uc_iwm"]},
      {"category":"chemical","text":"HRAC Group 1 denotes inhibition of ACCase; Group 2 denotes inhibition of ALS. These are mode-of-action classifications, not proof that a product will control a particular population or be safe in a crop.","source_ids":["hrac"]},
      {"category":"classification","text":"A weed is a plant growing where it is unwanted. Crop-versus-weed labels in the classifier describe the dataset classes; whether to retain or remove a plant depends on its context.","source_ids":["cleavers_rhs"]},
    ]
    payload={"schema_version":1,"curated_date":"2026-10-31","scope":"Actual dataset species only; source-checked regional information, not locally field-validated advice","species":records,"sources":sources,"general":general}
    path=ROOT/"knowledge_base/species.json";path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    review=[{"species":row["class_label"],"review_needed":["Confirm identification in local field conditions","Verify Pakistani crop relevance before deployment","Local agronomist review of any management application"]+(["Species-specific management record remains incomplete"] if row["tier"]=="B" and row["role"]=="weed" else [])} for row in records]
    (ROOT/"knowledge_base/BOTANICAL_REVIEW.json").write_text(json.dumps(review,indent=2),encoding="utf-8")
    (ROOT/"knowledge_base/README.md").write_text("""# Curated botanical knowledge base

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
""",encoding="utf-8")
    print(f"Curated {len(records)} species, {len(sources)} references, and {sum(len(r['claims']) for r in records)} species claims.")


if __name__=="__main__":build()
