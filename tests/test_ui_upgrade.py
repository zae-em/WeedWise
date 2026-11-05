"""Verify UI data operations, explicit saving, context and protected ML artifacts."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from streamlit.testing.v1 import AppTest
from app.views.species import filtered_species
from weed_assistant.config import ROOT
from weed_assistant.database import Database
from weed_assistant.knowledge import KnowledgeBase
from weed_assistant.chatbot import GroundedChatbot
from weed_assistant.inference import PlantClassifier
from weed_assistant.service import IdentificationService


class UIDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.db=Database(self.root/"history.sqlite3")
        self.kb=KnowledgeBase();self.db.sync_species(self.kb.records)

    def tearDown(self):self.temp.cleanup()

    def record(self,label="Maize",image="",confidence=.8):
        result={"predictions":[{"class_label":label,"confidence":confidence}],"needs_review":False,"crop_weed_label":"crop"}
        return self.db.save_prediction(image,"hash",result,{"usable":True},"model")

    def test_filters_stats_and_latest_feedback(self):
        first=self.record();second=self.record("Common wheat",confidence=.3)
        self.db.add_feedback(first,"Incorrect","Sugar beet","First review")
        self.db.add_feedback(first,"Unsure","Common wheat","Latest review")
        self.assertEqual(len(self.db.history(search="maiz")),1)
        self.assertEqual(len(self.db.history(search="%")),0)
        self.assertEqual(len(self.db.history(search="' OR 1=1 --")),0)
        self.assertEqual(self.db.history(species="Maize")[0]["corrected_class"],"Common wheat")
        self.assertEqual(len(self.db.history(low_confidence=.5)),1)
        self.assertEqual(len(self.db.history(date_from="2999-01-01")),0)
        self.assertEqual(len(self.db.history(date_to="2000-01-01")),0)
        self.assertEqual(self.db.stats()["predictions"],2)
        self.assertEqual(self.db.stats()["feedback"],2)
        self.assertEqual(set(self.db.all_prediction_ids()),{first,second})

    def test_deletion_removes_children_but_preserves_shared_photo(self):
        uploads=self.root/"uploads";uploads.mkdir();photo=uploads/"plant.jpg";photo.write_bytes(b"retained test image")
        first=self.record(image=str(photo));second=self.record(image=str(photo))
        self.db.add_feedback(first,"Incorrect","Common wheat")
        self.assertEqual(self.db.delete_predictions([first],uploads)["deleted"],1)
        self.assertFalse(self.db.feedback());self.assertIsNone(self.db.get_prediction(first));self.assertTrue(photo.exists())
        self.assertEqual(self.db.delete_predictions([second],uploads)["images_removed"],1)
        self.assertFalse(photo.exists());self.assertEqual(self.db.stats()["species_records"],len(self.kb.records))

    def test_deletion_never_unlinks_outside_uploads(self):
        outside=self.root/"original.jpg";outside.write_bytes(b"preserve")
        identifier=self.record(image=str(outside));self.db.delete_predictions([identifier],self.root/"uploads")
        self.assertTrue(outside.exists())

    def test_clearing_explicit_history_ids_preserves_species_and_drafts(self):
        for _ in range(3):self.record()
        self.db.add_draft("Maize","Automated temporary draft","https://example.org/test")
        self.assertEqual(self.db.delete_predictions(self.db.all_prediction_ids())["deleted"],3)
        self.assertFalse(self.db.history());self.assertEqual(self.db.stats()["species_records"],len(self.kb.records))
        self.assertEqual(len(self.db.drafts()),1)

    def test_filters_come_from_actual_species_records(self):
        self.assertEqual(len(filtered_species(self.kb.records,role="Weeds")),sum(r['role']=='weed' for r in self.kb.records))
        self.assertEqual(len(filtered_species(self.kb.records,role="Crops")),sum(r['role']=='crop' for r in self.kb.records))
        self.assertEqual(filtered_species(self.kb.records,search="Zea mays")[0]["class_label"],"Maize")
        self.assertTrue(all(r["family"]=="Poaceae" for r in filtered_species(self.kb.records,family="Poaceae")))
        self.assertFalse(filtered_species(self.kb.records,search="not-a-real-supported-species"))

    def test_chat_context_and_comparison_remain_grounded(self):
        bot=GroundedChatbot(self.kb)
        answer=bot.answer("How can I manage this weed without herbicides?","Common Chickweed")
        self.assertTrue(answer["supported"]);self.assertTrue(all(c["species"]=="Common Chickweed" for c in answer["evidence"]))
        comparison=bot.answer("Compare the identifying features of blackgrass and loose silky-bent.")
        self.assertEqual({c["species"] for c in comparison["evidence"]},{"Black-grass","Loose Silky-bent"})

    def test_explicit_save_no_photo_and_idempotency(self):
        classifier=PlantClassifier();service=IdentificationService(classifier,self.db,self.kb,self.root/"uploads")
        manifest=json.loads((ROOT/"dataset/processed/manifest.json").read_text())
        row=next(r for r in manifest["records"] if r["split"]=="test")
        data=(ROOT/"dataset/processed"/row["processed_path"]).read_bytes()
        before=service.identify(data,save=False,crop_context="Rice")
        self.assertIsNone(before["id"]);self.assertFalse(self.db.history())
        self.assertFalse(before["result"]["crop_context_applied_to_ranking"])
        original=classifier.predict(before["image"],before["quality"])
        self.assertEqual(before["result"]["predictions"],original["predictions"])
        identifier=service.save(before,retain_image=False)
        self.assertEqual(self.db.get_prediction(identifier)["image_path"],"")
        self.assertEqual(service.save(before),identifier);self.assertEqual(len(self.db.history()),1)

    def test_protected_artifacts_unchanged(self):
        baseline=json.loads((ROOT/"evaluation/ui/protected_artifacts_before.json").read_text())
        for name,expected in baseline.items():
            with (ROOT/name).open("rb") as stream:actual=hashlib.file_digest(stream,"sha256").hexdigest()
            self.assertEqual(actual,expected,name+" was modified")


class UIRoutingTests(unittest.TestCase):
    def test_dashboard_actions_and_query_navigation(self):
        app=AppTest.from_file(str(ROOT/"app/main.py"),default_timeout=60).run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label=="Identify a Plant").click().run()
        self.assertEqual(app.sidebar.radio[0].value,"Identify plant");self.assertFalse(app.exception)
        self.assertEqual(app.query_params["page"],"identify")

    def test_species_filter_and_empty_state(self):
        app=AppTest.from_file(str(ROOT/"app/main.py"),default_timeout=60)
        app.query_params["page"]="species";app.run();self.assertFalse(app.exception)
        next(box for box in app.selectbox if box.label=="Plant role").set_value("Crops").run()
        self.assertFalse(app.exception)
        next(box for box in app.text_input if box.label=="Search common or scientific name").set_value("no supported plant by this name").run()
        self.assertTrue(any("No supported species" in item.value for item in app.info))


if __name__=="__main__":unittest.main()
