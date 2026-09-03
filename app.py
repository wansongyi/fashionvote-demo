from __future__ import annotations

import base64
import io
import random
import threading
from collections import Counter
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from inference import CLASSES, FashionClassifier


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "static" / "demo_images"
MODEL_PATH = BASE_DIR / "model" / "fashionvote-resnet34-3class.pth"
DISPLAY_NAMES = {
    "black_skirt": "Black Skirt",
    "gray_coat": "Gray Coat",
    "white_skirt": "White Skirt",
}


def build_catalog() -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for category in CLASSES:
        directory = IMAGE_DIR / category
        catalog[category] = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not catalog[category]:
            raise RuntimeError(f"No demo images found for {category}")
    return catalog


def create_app(classifier: FashionClassifier | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        SEND_FILE_MAX_AGE_DEFAULT=3600,
    )

    catalog = build_catalog()
    model = classifier or FashionClassifier(MODEL_PATH)
    vote_counts: Counter[str] = Counter()
    vote_lock = threading.Lock()
    chooser = random.SystemRandom()

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'none'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    def image_item(category: str, filename: str) -> dict[str, str]:
        return {
            "category": category,
            "category_name": DISPLAY_NAMES[category],
            "filename": filename,
        }

    def choose_pair() -> tuple[dict[str, str], dict[str, str]]:
        categories = chooser.sample(list(CLASSES), 2)
        return tuple(
            image_item(category, chooser.choice(catalog[category]))
            for category in categories
        )

    def validate_item(value: str) -> tuple[str, str]:
        try:
            category, filename = value.split("/", 1)
        except ValueError:
            abort(400)
        if category not in catalog or filename not in catalog[category]:
            abort(400)
        return category, filename

    @app.get("/")
    def landing():
        featured = [image_item(category, catalog[category][0]) for category in CLASSES]
        return render_template("index.html", featured=featured)

    @app.route("/vote", methods=["GET", "POST"])
    def vote():
        message = None
        if request.method == "POST":
            left = validate_item(request.form.get("left", ""))
            right = validate_item(request.form.get("right", ""))
            chosen = validate_item(request.form.get("choice", ""))
            if chosen not in {left, right}:
                abort(400)
            with vote_lock:
                vote_counts[chosen[0]] += 1
            message = f"Vote counted for {DISPLAY_NAMES[chosen[0]]}. Thanks for shaping the demo trend!"

        left_item, right_item = choose_pair()
        with vote_lock:
            totals = {category: vote_counts[category] for category in CLASSES}
        return render_template(
            "vote.html",
            left=left_item,
            right=right_item,
            totals=totals,
            message=message,
        )

    @app.route("/recommend", methods=["GET", "POST"])
    def recommend():
        if request.method == "GET":
            samples = [image_item(category, catalog[category][1]) for category in CLASSES]
            return render_template("recommend.html", samples=samples)

        upload = request.files.get("image")
        sample = request.form.get("sample", "")
        raw: bytes

        if upload and upload.filename:
            raw = upload.read()
        elif sample:
            category, filename = validate_item(sample)
            raw = (IMAGE_DIR / category / filename).read_bytes()
        else:
            return render_template("recommend.html", samples=[], error="Choose an image to analyze."), 400

        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
            if image.width * image.height > 20_000_000:
                raise ValueError("Image dimensions are too large")
            prediction = model.predict(image)
        except (UnidentifiedImageError, OSError, ValueError):
            return render_template("recommend.html", samples=[], error="That file is not a valid supported image."), 400

        preview_buffer = io.BytesIO()
        preview_image = image.convert("RGB")
        preview_image.thumbnail((900, 900))
        preview_image.save(preview_buffer, format="JPEG", quality=84)
        preview = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")
        recommendations = [
            image_item(prediction.label, filename)
            for filename in chooser.sample(catalog[prediction.label], min(3, len(catalog[prediction.label])))
        ]
        return render_template(
            "recommendation_result.html",
            prediction=prediction,
            category_name=DISPLAY_NAMES[prediction.label],
            preview=preview,
            recommendations=recommendations,
            display_names=DISPLAY_NAMES,
        )

    @app.get("/images/<category>/<filename>")
    def demo_image(category: str, filename: str):
        validate_item(f"{category}/{filename}")
        return send_from_directory(IMAGE_DIR / category, filename)

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", model="resnet34-3class", classes=list(CLASSES))

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="That runway does not exist."), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", code=413, message="Please choose an image smaller than 5 MB."), 413

    return app


app = create_app()
