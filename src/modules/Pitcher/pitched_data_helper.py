"""Pitched data helper module"""


def get_frequencies_with_high_confidence(
    frequencies: list[float], confidences: list[float], threshold=0.4
) -> tuple[list[float], list[float]]:
    """Get frequencies with high confidence and their corresponding weights.

    Filters frequencies by a confidence threshold. If no frequency passes
    the threshold, falls back to the top 25% by confidence instead of
    returning all frequencies (which would include low-confidence noise).

    Args:
        frequencies: List of detected frequencies (Hz).
        confidences: List of confidence values (0.0-1.0) for each frequency.
        threshold: Minimum confidence to accept a frequency (default 0.4).

    Returns:
        Tuple of (filtered_frequencies, confidence_weights).
    """
    conf_f = []
    conf_weights = []
    for i, conf in enumerate(confidences):
        if conf > threshold:
            conf_f.append(frequencies[i])
            conf_weights.append(conf)

    if not conf_f:
        # Fallback: use top 25% by confidence instead of ALL frequencies.
        # This avoids flooding the result with low-confidence noise frames.
        if confidences:
            indexed = list(enumerate(confidences))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_n = max(1, len(indexed) // 4)
            for idx, conf in indexed[:top_n]:
                conf_f.append(frequencies[idx])
                conf_weights.append(conf)

    return conf_f, conf_weights
