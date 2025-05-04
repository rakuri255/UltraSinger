"""Pitched data helper module"""
def get_frequencies_with_high_confidence(
    frequencies: list[float], confidences: list[float], threshold=0.4
) -> list[float]:
    """Get frequency with high confidence"""
    if not frequencies or not confidences:
        return []
    
    conf_f = []
    for i, (freq, conf) in enumerate(zip(frequencies, confidences)):
        if conf > threshold and freq > 0:
            conf_f.append(freq)
    
    # If no frequencies pass the confidence threshold, find the one with the highest confidence
    if not conf_f and frequencies and confidences: # Ensure lists are not empty
        max_conf_idx = -1
        max_conf = -1.0
        for i, conf in enumerate(confidences):
            # Check frequency is positive and confidence is highest so far
            if frequencies[i] > 0 and conf > max_conf:
                max_conf = conf
                max_conf_idx = i
        
        if max_conf_idx != -1: # Check if a valid frequency was found
            conf_f.append(frequencies[max_conf_idx])
            
    return conf_f