from __future__ import annotations

import io
import unittest
from pathlib import Path

from PIL import Image

from app import IMAGE_DIR, app
from inference import CLASSES, FashionClassifier


class FashionVoteSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_landing_and_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Style, decided", response.data)
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(tuple(health.json["classes"]), CLASSES)

    def test_vote_page_images_and_both_buttons(self):
        response = self.client.get("/vote")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertEqual(html.count('name="choice"'), 2)
        self.assertEqual(html.count("<img "), 2)

        values = []
        for marker in ('name="left" value="', 'name="right" value="'):
            values.append(html.split(marker, 1)[1].split('"', 1)[0])
        for choice in values:
            submitted = self.client.post("/vote", data={"left": values[0], "right": values[1], "choice": choice})
            self.assertEqual(submitted.status_code, 200)
            self.assertIn(b"Vote counted", submitted.data)

        category, filename = values[0].split("/", 1)
        image = self.client.get(f"/images/{category}/{filename}")
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.content_type.startswith("image/"))
        image.close()

    def test_prediction_route_and_probabilities(self):
        sample = next((IMAGE_DIR / CLASSES[0]).glob("*.jpg"))
        with sample.open("rb") as handle:
            response = self.client.post(
                "/recommend",
                data={"image": (io.BytesIO(handle.read()), "sample.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MODEL PREDICTION", response.data)
        self.assertIn(b"sum to 100%", response.data)

    def test_public_routes_do_not_error(self):
        for path in ("/", "/vote", "/recommend", "/healthz", "/missing"):
            response = self.client.get(path)
            self.assertNotEqual(response.status_code, 500, path)

    def test_model_predictions_are_three_class(self):
        classifier = FashionClassifier(Path(app.root_path) / "model" / "fashionvote-resnet34-3class.pth")
        for category in CLASSES:
            for sample in list((IMAGE_DIR / category).glob("*.jpg"))[:3]:
                with Image.open(sample) as image:
                    prediction = classifier.predict(image)
                self.assertEqual(set(prediction.probabilities), set(CLASSES))
                self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=5)
                self.assertIn(prediction.label, CLASSES)


if __name__ == "__main__":
    unittest.main()
