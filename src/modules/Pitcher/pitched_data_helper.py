"""Pitched data helper module"""
def get_frequencies_with_high_confidence(
    frequencies: list[float], confidences: list[float], threshold=0.4
) -> list[float]:
    """Get frequency with high confidence"""
    conf_f = []
    for i, conf in enumerate(confidences):
        if conf > threshold:
            conf_f.append(frequencies[i])
    if not conf_f:
        conf_f = frequencies
    return conf_f