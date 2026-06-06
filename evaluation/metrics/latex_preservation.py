"""
Metric 1: LaTeX/math segment preservation score.

Score = (number of math segments from source that appear identically in translation)
        / (total number of math segments in source)

Math segments: all $...$ and $$...$$ spans. $$...$$ is matched first to avoid
splitting double-dollar signs.

Currency amounts like $1,000 are excluded via a negative lookahead that rejects
$ followed by digits then a comma (e.g. $1,000), while still allowing math that
starts with a digit (e.g. $3x+2$).
"""
import re

_MATH_RE = re.compile(r'\$\$[\s\S]*?\$\$|\$(?!\d+[,\s])[^$\n]+?\$')


def extract_math_segments(text: str) -> list[str]:
    return _MATH_RE.findall(text)


def _normalize(text: str) -> str:
    text = text.replace(r"\_", "_")
    # add braces to bare multi-character subscripts: a_11 -> a_{11}
    text = re.sub(r'_(\w{2,})', r'_{\1}', text)
    return text


def latex_preservation_score(source_en: str, translation_pl: str) -> float:
    segments = extract_math_segments(source_en)
    if not segments:
        return 1.0
    translation_norm = _normalize(translation_pl)
    preserved = sum(1 for seg in segments if _normalize(seg) in translation_norm)
    return preserved / len(segments)


def score_batch(sources: list[str], translations: list[str]) -> float:
    if not sources:
        return 1.0
    scores = [latex_preservation_score(s, t) for s, t in zip(sources, translations)]
    return sum(scores) / len(scores)
