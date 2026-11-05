"""Conservative species resolution with saved GBIF name and vernacular evidence.

Broad, conflicting, or unverified labels stay excluded; stages and cultivar forms
are never advertised as separate species. Existing scientific mappings are retained.
"""
import concurrent.futures
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset/cropandweed/taxonomy'
LABELS_URL = 'https://raw.githubusercontent.com/cropandweed/cropandweed-dataset/main/cnw/utilities/datasets.py'
# Candidate associations are accepted only when GBIF vernacular evidence confirms them.
CANDIDATES = {
 'Maize': 'Zea mays', 'Sugar beet': 'Beta vulgaris', 'Pea': 'Pisum sativum',
 'Courgette': 'Cucurbita pepo', 'Radish': 'Raphanus sativus', 'Asparagus': 'Asparagus officinalis',
 'Potato': 'Solanum tuberosum', 'Flat leaf parsley': 'Petroselinum crispum',
 'Curly leaf parsley': 'Petroselinum crispum', 'Hemp': 'Cannabis sativa',
 'Sunflower': 'Helianthus annuus', 'Sage': 'Salvia officinalis', 'Common bean': 'Phaseolus vulgaris',
 'Faba bean': 'Vicia faba', 'Hybrid goosefoot': 'Chenopodium hybridum',
 'Black-bindweed': 'Fallopia convolvulus', 'Cockspur grass': 'Echinochloa crus-galli',
 'Red-root amaranth': 'Amaranthus retroflexus', 'White goosefoot': 'Chenopodium album',
 'Thorn apple': 'Datura stramonium', 'Potato weed': 'Galinsoga parviflora',
 'German chamomile': 'Matricaria chamomilla', 'Creeping thistle': 'Cirsium arvense',
 'Field milk thistle': 'Sonchus arvensis', 'Purslane': 'Portulaca oleracea',
 'Black nightshade': 'Solanum nigrum', 'Pale persicaria': 'Persicaria lapathifolia',
 'Cleavers': 'Galium aparine', 'Whitetop': 'Lepidium draba', 'Frosted orach': 'Atriplex sagittata',
 'Black horehound': 'Ballota nigra', 'Shepherds purse': 'Capsella bursa-pastoris',
 'Field bindweed': 'Convolvulus arvensis', 'Common mugwort': 'Artemisia vulgaris',
 'Hedge mustard': 'Sisymbrium officinale', 'Groundsel': 'Senecio vulgaris',
 'Broadleaf plantain': 'Plantago major', 'White ball-mustard': 'Calepina irregularis',
 'Peppermint': 'Mentha x piperita', 'Field pennycress': 'Thlaspi arvense',
 'Corn spurry': 'Spergula arvensis', 'Purple crabgrass': 'Digitaria sanguinalis',
 'Common fumitory': 'Fumaria officinalis', 'Ivy-leaved speedwell': 'Veronica hederifolia',
 'Annual meadow grass': 'Poa annua', 'Redshank': 'Persicaria maculosa',
 'Common hemp-nettle': 'Galeopsis tetrahit', 'Rough meadow-grass': 'Poa trivialis',
 'Green bristlegrass': 'Setaria viridis', 'Small geranium': 'Geranium pusillum',
 'Cornflower': 'Centaurea cyanus', 'Common corn-cockle': 'Agrostemma githago',
 'Creeping crowfoot': 'Ranunculus repens', 'Wall barley': 'Hordeum murinum',
 'Annual fescue': 'Vulpia myuros', 'Purple dead-nettle': 'Lamium purpureum',
 'Ribwort plantain': 'Plantago lanceolata', 'Pineappleweed': 'Matricaria discoidea',
 'Common chickweed': 'Stellaria media', 'Soft brome': 'Bromus hordeaceus',
 'Wild pansy': 'Viola tricolor', 'Yellow rocket': 'Barbarea vulgaris',
 'Common wild oat': 'Avena fatua', 'Red poppy': 'Papaver rhoeas', 'Rye brome': 'Bromus secalinus',
 'Knotgrass': 'Polygonum aviculare', 'Prickly lettuce': 'Lactuca serriola',
 'Copse-bindweed': 'Fallopia dumetorum', 'Common buckwheat': 'Fagopyrum esculentum',
 'Chives': 'Allium schoenoprasum', 'Garlic': 'Allium sativum', 'Soybean': 'Glycine max',
 'Wild carrot': 'Daucus carota', 'Common horsetail': 'Equisetum arvense',
 'Common dandelion': 'Taraxacum officinale'
}


def norm(value):
    return re.sub('[^a-z]', '', value.lower())


def fetch(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return json.load(response)


def resolve(pair):
    label, candidate = pair
    path = OUT / (norm(label) + '.json')
    if path.exists():
        return json.loads(path.read_text())
    url = 'https://api.gbif.org/v1/species/match?' + urllib.parse.urlencode({'name': candidate, 'strict': 'true'})
    result = {'label': label, 'candidate': candidate, 'match_url': url, 'verified': False}
    try:
        match = fetch(url)
        result['match'] = match
        key = match.get('acceptedUsageKey', match.get('usageKey'))
        if match.get('matchType') != 'EXACT' or match.get('rank') != 'SPECIES' or not key:
            result['reason'] = 'No exact species-rank match'
        else:
            result['accepted'] = fetch(f'https://api.gbif.org/v1/species/{key}')
            vern_url = f'https://api.gbif.org/v1/species/{key}/vernacularNames?limit=1000'
            result['vernacular_url'] = vern_url
            result['vernaculars'] = fetch(vern_url)
            names = {norm(v['vernacularName']) for v in result['vernaculars']['results']}
            result['verified'] = norm(label) in names
            result['reason'] = 'Scientific name and common-label association verified' if result['verified'] else 'Common-label association needs review'
    except Exception as error:
        result['reason'] = type(error).__name__ + ': ' + str(error)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / 'dataset/cropandweed/metadata.json').read_text())
    old = json.loads((ROOT / 'knowledge_base/species.json').read_text())['species']
    # The original sugar-beet record names a cultivated variety. Resolve overlap
    # at species rank so it cannot become a second Beta vulgaris output.
    old_science = {norm(' '.join(r['scientific_name'].split()[:2])): r for r in old}
    old_science.update({norm(r['scientific_name']): r for r in old})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        evidence = {r['label']: r for r in pool.map(resolve, CANDIDATES.items())}
    mappings = {}
    for raw_id, raw_label in metadata['mask_targets']['segmentation'].items():
        base = re.sub(r' (two|four|six|eight)-leaf stage$| [Mm]ax$', '', raw_label)
        ev = evidence.get(base, {})
        item = {'raw_id': int(raw_id), 'raw_label': raw_label, 'base_label': base,
                'included': False, 'reason': ev.get('reason', 'Generic, ambiguous, or unresolved label')}
        if ev.get('verified'):
            accepted = ev['accepted']
            scientific = accepted.get('canonicalName', ev['candidate'])
            existing = old_science.get(norm(scientific)) or old_science.get(norm(ev['candidate']))
            # Crop role follows the authors' explicit CropOrWeed2 crop IDs.
            crop_ids = {1,2,3,4,5,6,7,8,9,10,11,12,13,15,18,24,26,27,94}
            weed_ids = {31,48,62,65,68,69,74,75,81,84,86,32,29,33,37,49,30,44,66,87,89,91,61,79,34,41,52,35,36,78,38,39,71,72,88,42,45,70,47,51,54,58,60,80,83,96,22,63,85,56,57,64,77,50,59,67,76}
            role = existing['role'] if existing else ('crop' if int(raw_id) in crop_ids else 'weed' if int(raw_id) in weed_ids else None)
            item.update(scientific_name=scientific, family=accepted.get('family', 'Unresolved'),
                        class_label=existing['class_label'] if existing else scientific,
                        common_name=base, role=role, gbif_key=accepted['key'],
                        source_urls=[ev['match_url'], ev['vernacular_url'], LABELS_URL])
            item['included'] = role is not None
            if role is None:
                item['reason'] = 'Species verified; authors simplified crop/weed role does not cover this label'
        mappings[raw_id] = item
    target = ROOT / 'dataset/cropandweed/taxonomy.json'
    target.write_text(json.dumps({'schema_version': 1, 'labels': mappings,
        'policy': 'Exact GBIF species match plus exact normalized common name; author-defined crop/weed role; ambiguous labels excluded',
        'evidence_directory': 'taxonomy'}, indent=2), encoding='utf-8')
    print('Resolved raw labels:', sum(r['included'] for r in mappings.values()), 'of', len(mappings), flush=True)


if __name__ == '__main__':
    main()
