"""Meaningful expansion checks; fixtures do not stand in for trained v2 metrics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.expansion_data import BKTree, assign_groups, box_pixels, extract_instance
from training.expanded_train import BalancedCoverageSampler, expand_model, load_manifest, train, ExpansionConfig
from weed_assistant.config import ROOT


class ExpansionDataTests(unittest.TestCase):
    def test_near_duplicate_index(self):
        tree = BKTree()
        for i,value in enumerate([0,1,3,255]):tree.insert(value,i)
        self.assertEqual(set(tree.find(0,2)),{0,1,2})

    def test_frozen_heldout_propagates_to_entire_new_session(self):
        rows=[{'source_image':'old','session_id':'old-g','origin':'original','class_name':'a','phash':'0000000000000000','split':'test'},
              {'source_image':'new1','session_id':'new-session','origin':'cropandweed','class_name':'a','phash':'0000000000000001'},
              {'source_image':'new2','session_id':'new-session','origin':'cropandweed','class_name':'b','phash':'ffffffffffffffff'}]
        assign_groups(rows)
        self.assertEqual({r['split'] for r in rows},{'test'})

    def test_conflicting_frozen_splits_quarantine_new_session(self):
        rows=[{'source_image':'old1','session_id':'g1','origin':'original','class_name':'a','phash':'0000000000000000','split':'test'},
              {'source_image':'old2','session_id':'g2','origin':'original','class_name':'a','phash':'0000000000000001','split':'train'},
              {'source_image':'new','session_id':'g3','origin':'cropandweed','class_name':'a','phash':'0000000000000003'}]
        rejected=assign_groups(rows)
        self.assertEqual(len(rejected),1)
        self.assertEqual(rows[0]['split'],'test')
        self.assertEqual(rows[1]['split'],'train')

    def test_semantic_crop_and_overlap_rejection(self):
        image=Image.new('RGB',(100,100),'green')
        mask=np.zeros((100,100),dtype=np.uint8);mask[25:75,25:75]=1
        detection={'label':'Maize','label_id':1,'bounding_box':[.2,.2,.6,.6]}
        taxonomy={'1':{'included':True,'raw_label':'Maize','scientific_name':'Zea mays'}}
        crop,_,box,_=extract_instance(image,mask,detection,[detection],taxonomy)
        self.assertEqual(crop.size,(224,224));self.assertEqual(box,(20,20,80,80))
        with self.assertRaisesRegex(ValueError,'overlapping'):
            extract_instance(image,mask,detection,[detection,copy.deepcopy(detection)],taxonomy)
        mask[25:50,25:75]=2
        with self.assertRaisesRegex(ValueError,'mixed_species'):
            extract_instance(image,mask,detection,[detection],taxonomy)

    def test_invalid_boxes_are_rejected(self):
        for box in [[0,0,-1,1],[.9,.9,.2,.2],[0,0,float('nan'),1]]:
            with self.assertRaises(ValueError):box_pixels(box,100,100)

    def test_growth_stages_share_scientific_species(self):
        taxonomy=json.loads((ROOT/'dataset/cropandweed/taxonomy.json').read_text())['labels']
        for ids in [range(1,7),range(7,13)]:
            self.assertEqual(len({taxonomy[str(i)]['scientific_name'] for i in ids}),1)
        for i in ['0','255']:
            self.assertFalse(taxonomy[i]['included'])
        self.assertEqual(taxonomy['7']['class_label'],'Sugar beet')


class ExpansionModelTests(unittest.TestCase):
    def test_model_version_history_preserves_mapping_and_predictions(self):
        from weed_assistant.database import Database
        with tempfile.TemporaryDirectory() as folder:
            database=Database(Path(folder)/'history.sqlite3')
            database.register_model('old','v1',['a','b'],{'a':'weed','b':'crop'})
            result={'predictions':[{'class_label':'a','confidence':.8}],'needs_review':False,'model_version':'v1'}
            identifier=database.save_prediction('','image',result,{},'old')
            database.register_model('new','v2',['a','b','c'],{'a':'weed','b':'crop','c':'weed'})
            self.assertEqual(database.get_prediction(identifier)['model_sha256'],'old')
            self.assertEqual(json.loads(database.get_prediction(identifier)['result_json'])['model_version'],'v1')
            with self.assertRaises(ValueError):database.register_model('old','v1',['b','a'],{'a':'weed','b':'crop'})

    def test_expanded_artifact_without_roles_is_rejected(self):
        from weed_assistant.inference import PlantClassifier,ModelUnavailable
        artifact=torch.load(ROOT/'models/mobilenet_v2/model.pt',map_location='cpu',weights_only=True)
        artifact['model_version']='v2';artifact.pop('class_roles',None)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';torch.save(artifact,path)
            with self.assertRaisesRegex(ModelUnavailable,'explicit crop/weed'):PlantClassifier(path)

    def test_balanced_sampler_keeps_complete_split(self):
        labels=[0]*20+[1]*2+[2]*10
        sampler=BalancedCoverageSampler(labels,torch.Generator().manual_seed(42))
        order=list(sampler)
        self.assertEqual(set(order),set(range(len(labels))))
        self.assertEqual(len(order),len(sampler))
        self.assertLessEqual(sum(labels[i]==1 for i in order),6)

    def test_original_backbone_and_head_rows_preserved_with_reordered_classes(self):
        torch.set_num_threads(2)
        artifact=torch.load(ROOT/'models/mobilenet_v2/model.pt',map_location='cpu',weights_only=True)
        classes=['New species']+list(reversed(artifact['classes']))
        expanded,teacher=expand_model(artifact,classes)
        self.assertTrue(torch.equal(expanded.features[0][0].weight,teacher.features[0][0].weight))
        for old,name in enumerate(artifact['classes']):
            self.assertTrue(torch.equal(expanded.classifier[1].weight[classes.index(name)],teacher.classifier[1].weight[old]))
        expanded.eval()
        with torch.inference_mode():logits=expanded(torch.zeros(1,3,224,224))
        self.assertEqual(logits.shape,(1,13));self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipIf(torch.cuda.is_available(),'This guard specifically verifies CPU hosts')
    def test_no_cpu_training_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'CUDA GPU'):
            train(ExpansionConfig())


@unittest.skipUnless((ROOT/'models/v2/model.pt').is_file() and (ROOT/'dataset/combined_v2/manifest.json').is_file(),
                     'Genuine trained v2 artifact and combined test data are not available yet')
class RealExpandedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from weed_assistant.inference import PlantClassifier
        cls.classifier=PlantClassifier(ROOT/'models/v2/model.pt')
        cls.manifest,cls.splits=load_manifest(ROOT/'dataset/combined_v2')

    def test_real_expanded_mapping_and_roles(self):
        self.assertEqual(self.classifier.classes,self.manifest['classes'])
        self.assertTrue(set(self.manifest['original_classes'])<set(self.classifier.classes))
        self.assertEqual(set(self.classifier.class_roles),set(self.classifier.classes))

    def test_real_original_and_new_species_inference(self):
        with np.load(ROOT/'evaluation/v2/test_predictions.npz') as saved:
            correct=saved['probabilities'].argmax(1)==saved['labels']
        for origin in ['original','cropandweed']:
            index=next(i for i,r in enumerate(self.splits['test']) if r['origin']==origin and correct[i] and
                       (origin=='original' or r['class_name'] not in self.manifest['original_classes']))
            row=self.splits['test'][index]
            with Image.open(row['path']) as image:result=self.classifier.predict(image)
            self.assertEqual(result['predictions'][0]['class_label'],row['class_name'])
            self.assertEqual(len(result['predictions']),5)
            self.assertAlmostEqual(result['crop_probability']+result['weed_probability'],1,places=6)

    def test_matching_grounded_expanded_knowledge(self):
        from weed_assistant.knowledge import KnowledgeBase
        from weed_assistant.chatbot import GroundedChatbot
        kb=KnowledgeBase(ROOT/'knowledge_base/species_v2.json')
        self.assertEqual(set(kb.by_class),set(self.classifier.classes))
        label=next(c for c in self.classifier.classes if c not in self.manifest['original_classes'])
        self.assertTrue(GroundedChatbot(kb).answer('What is the scientific name of '+label+'?')['supported'])


if __name__ == '__main__':unittest.main()
