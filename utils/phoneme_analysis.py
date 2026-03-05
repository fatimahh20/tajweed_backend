import re

# Cell 1: Helper for IPA visualization
def get_friendly_symbol(ipa_symbol):
    mapping = {
        'ð': 'th', 'θ': 'th_unvoiced', 'ʃ': 'sh',
        'χ': 'kh', 'ħ': 'H', 'ʕ': 'ʕ',
        'aː': 'A (long)', 'iː': 'Y (long)', 'uː': 'W (long)'
    }
    return mapping.get(ipa_symbol, ipa_symbol)

# Cell 4 Constants
EQUIVALENCE_MAP = {
    'a': ['a', 'A', 'aː'], 'i': ['i', 'Y', 'iː'], 
    'u': ['u', 'W', 'uː'], 
}

FEEDBACK_RULES = {
    ("ħ", "h"): "Practise HuruF E Halqi properly .",
    ("h", "ħ"): "Practise HuruF E Halqi properly .",
    ("aː", "ʕ"): "Practise HuruF E Halqi properly.",
    ("ʕ", "aː"): "Alif without Halaq.",
    ("q", "k"): " ق should be Heavy.",
    ("k", "q"): "ک should be light.",
    ("s", "θ"): " س with whistle sound.",
    ("θ", "s"): " ث from tounge tip.",
    ("s̪", "θ"): " ص with rolled lips.",
    ("θ", "s̪"): " ث from tounge tip.",
    ("s̪", "s"): " ص with rolled lips.",
    ("s", "s̪"): " س with whistle sound.",

    ("ð", "z"): "Practise Thaal (ذ) vs. Zay (ز) properly (soft 'th' vs. buzzing 'z').",
    ("z", "ð"): "Practise Thaal (ذ) vs. Zay (ز) properly (buzzing 'z' vs. soft 'th').",
    ("ð", "dˤ"): "Practise Thaal (ذ) vs. Daad (ض) properly (soft 'th' vs. emphatic 'd').",
    ("dˤ", "ð"): "Practise Thaal (ذ) vs. Daad (ض) properly (emphatic 'd' vs. soft 'th').",
    ("ð", "ðˤ"): "Practise Thaal (ذ) vs. Zaa (ظ) properly (soft 'th' vs. emphatic 'dh').",
    ("ðˤ", "ð"): "Practise Thaal (ذ) vs. Zaa (ظ) properly (emphatic 'dh' vs. soft 'th').",
    ("z", "dˤ"): "Practise Zay (ز) vs. Daad (ض) properly (buzzing 'z' vs. emphatic 'd').",
    ("dˤ", "z"): "Practise Zay (ز) vs. Daad (ض) properly (emphatic 'd' vs. buzzing 'z').",
    ("z", "ðˤ"): "Practise Zay (ز) vs. Zaa (ظ) properly (buzzing 'z' vs. emphatic 'dh').",
    ("ðˤ", "z"): "Practise Zay (ز) vs. Zaa (ظ) properly (emphatic 'dh' vs. buzzing 'z').",
    ("dˤ", "ðˤ"): "Practise Daad (ض) vs. Zaa (ظ) properly (emphatic 'd' vs. emphatic 'dh').",
    ("ðˤ", "dˤ"): "Practise Daad (ض) vs. Zaa (ظ) properly (emphatic 'dh' vs. emphatic 'd')."
}

CONFUSION_MAP = {
    "s": ["s̪", "θ"],
    "s̪": ["s","θ"],
    "k": ["q"],
    "q": ["k"],
    "θ": ["s", "s̪"],
    "h": ["ħ"],
    "ħ": ["h"],
    "t": ["t̪"],
    "t̪": ["t"],
    "aː": ["ʕ"],
    "ʕ": ["aː"],
    "ð": ["dˤ","z","ðˤ"],
    "z": ["dˤ","ð","ðˤ"],
    "dˤ": ["ð","z","ðˤ"],
    "aː": ["aʔaː",],
    "aʔaː":["aː"],
    "aː": ["a"],
    "a":["aː"],
}




ARABIC_TO_PHONEME = {
    "ع": "ʕ",
    "ا": "aː",
    "ب": "b",
    "ت": "t",
    "ث": "θ",
    "ج": "dʒ",
    "ح": "ħ",
    "خ": "x",
    "د": "d",
    "ذ": "ð",
    "ر": "ɹ",
    "ز": "z",
    "س": "s",
    "ش": "ʃ",
    "ص": "s̪",
    "ض": "dˤ",
    "ط": "t̪",
    "ظ": "ðˤ",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "j",
    "ء": "ʔ", # Added glottal stop for consistency with metadata

    # Harakat
    "َ": "a",
    "ِ": "i",
    "ُ": "u",
    "ً": "an",
    "ٍ": "in",
    "ٌ": "un",
    "~": "aʔaː",

}

PHONEME_TO_ARABIC = {v: k for k, v in ARABIC_TO_PHONEME.items()}

def phoneme_timestamps(predicted_ids, num_frames, audio_len, processor):
    """EXACT Timestamp function from your Colab"""
    frame_duration = audio_len / num_frames
    result = []
    vocab_inv = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    blank_token_id = processor.tokenizer.pad_token_id
    current_segment_token_id = -1
    segment_start_frame = 0

    for i, token_id in enumerate(predicted_ids):
        if token_id == blank_token_id:
            if current_segment_token_id != -1:
                token_str = vocab_inv[current_segment_token_id].replace('|', '').strip()
                if token_str:
                    result.append((token_str, segment_start_frame * frame_duration, i * frame_duration))
            current_segment_token_id = -1
        elif token_id != current_segment_token_id:
            if current_segment_token_id != -1:
                token_str = vocab_inv[current_segment_token_id].replace('|', '').strip()
                if token_str:
                    result.append((token_str, segment_start_frame * frame_duration, i * frame_duration))
            current_segment_token_id = token_id
            segment_start_frame = i
    return result

def is_equivalent(ref, detected):
    if ref == detected: return True
    for variants in EQUIVALENCE_MAP.values():
        if ref in variants and detected in variants: return True
    return False
def get_full_analysis(expected_str, predicted_ids, frames, audio_len, processor):
    # 1. Generate Raw Timestamps (Your Cell 2 logic)
    raw_user_phonemes = phoneme_timestamps(predicted_ids, frames, audio_len, processor)
    
    # 2. Filter segments > 0.01s (Your Cell 3 logic)
    detected_for_acc = [token for token, start, end in raw_user_phonemes if (end - start) > 0.01]

    # 3. Accuracy Calculation (Your Cell 4 logic)
    expected = list(expected_str.replace(" ", ""))
    matches = sum(1 for i in range(min(len(expected), len(detected_for_acc))) 
                  if is_equivalent(expected[i], detected_for_acc[i]))
    accuracy = (matches / len(expected)) * 100 if expected else 0

    # 4. Substitution Feedback (Your Cell 4 logic integrated)
    feedback_strings = []
    det_set = set(detected_for_acc)
    
    for ref_ph in expected:
        # If the phoneme was pronounced correctly, skip feedback
        if any(is_equivalent(ref_ph, d) for d in detected_for_acc): 
            continue
            
        # Check for common confusions (Cell 4 logic)
        confused = CONFUSION_MAP.get(ref_ph, [])
        sub = next((u for u in det_set if u in confused), None)
        
        if sub:
            specific_message = FEEDBACK_RULES.get((ref_ph, sub))
            arabic_sub = PHONEME_TO_ARABIC.get(sub, sub)
            arabic_ref = PHONEME_TO_ARABIC.get(ref_ph, ref_ph)
            
            # Convert dictionary-style data into a single string
            if specific_message:
                feedback_strings.append(f'"{specific_message}" - Used "{arabic_sub}" instead of "{arabic_ref}"')
            else:
                feedback_strings.append(f'Used "{arabic_sub}" instead of "{arabic_ref}"')

    # RETURN EXACTLY 4 VALUES: accuracy, feedback_list, expected_list, detected_list
    # This solves the "expected 4, got 5" error
    return round(accuracy, 2), feedback_strings, expected, detected_for_acc