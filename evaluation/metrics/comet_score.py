"""
Metric 2: COMET quality estimation score.

Uses Unbabel/wmt22-cometkiwi-da — a reference-free QE model (~1.3 GB) that scores
translations from source + MT only (no human reference needed).

Install: pip install unbabel-comet
"""


def load_comet_model(model_name: str = "Unbabel/wmt22-cometkiwi-da"):
    from comet import download_model, load_from_checkpoint
    from huggingface_hub.errors import GatedRepoError
    print(f"Loading COMET model {model_name} ...")
    try:
        path = download_model(model_name)
    except GatedRepoError:
        raise SystemExit(
            f"\n[ERROR] Access to {model_name} is restricted.\n"
            f"Go to https://huggingface.co/{model_name} login and agree to terms of use',\n"
            f""
        )
    return load_from_checkpoint(path)


def score_batch(
    sources: list[str],
    translations: list[str],
    model,
    batch_size: int = 8,
) -> list[float]:
    """Return per-example COMET scores."""
    data = [{"src": s, "mt": t} for s, t in zip(sources, translations)]
    output = model.predict(data, batch_size=batch_size, gpus=0)
    return list(output.scores)


def comet_score(
    sources: list[str],
    translations: list[str],
    model=None,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
) -> float:
    """Return mean COMET score over all (source, translation) pairs."""
    if model is None:
        model = load_comet_model(model_name)
    scores = score_batch(sources, translations, model)
    return sum(scores) / len(scores) if scores else 0.0
