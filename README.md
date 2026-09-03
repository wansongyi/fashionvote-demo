# FashionVote Demo

A sanitized, database-free portfolio demo recovered from the original FashionVote project.

## Public-demo architecture

- Flask and Gunicorn
- 108 bundled fashion images from the recovered collection
- read-only, three-class ResNet34 checkpoint
- anonymous in-memory voting
- in-memory image prediction; uploaded files are never persisted
- no database, accounts, external AI API, or application secrets

## Local run

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
gunicorn --bind 127.0.0.1:8080 --workers 1 --threads 4 app:app
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Model provenance

The recovered checkpoint had a five-output classifier because the recovered training script explicitly replaced the ResNet34 head with `Linear(512, 5)`. It contained no class metadata, while the recovered public dataset has exactly three label directories and `class_indices.json` maps only three labels.

For this demo, the recovered convolutional backbone was loaded without its incompatible head. A new three-output head and the final residual block were fine-tuned on the 431 recovered images across `black_skirt`, `gray_coat`, and `white_skirt`, using a deterministic 80/20 stratified split. The resulting versioned checkpoint embeds its three class names and is rejected by the app if its head or label metadata does not match.

The selected checkpoint achieved 86/86 correct predictions on the held-out split (29 black skirts, 29 gray coats, and 28 white skirts). This is a recovery-set validation result, not a claim of general fashion-model accuracy.

The original five-output checkpoint and any recovered credentials are intentionally excluded.
