""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"" ⟨⟩'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻᴀᴄᴇᴋᴍᴏᴘᴜᴡꜰꜱ"
)

# Special vocal/non-speech tokens using distinctive Unicode symbols
_special_vocals = [
    "⟨ᴜʜ⟩",        # <UH> - hesitation filler
    "⟨ᴜᴍ⟩",        # <UM> - hesitation filler  
    "⟨ʟᴀᴜɢʜ⟩",     # <LAUGH> - laughter
    "⟨ɢɪɢɢʟᴇ⟩",    # <GIGGLE> - light laughter
    "⟨ᴄʜᴜᴄᴋʟᴇ⟩",   # <CHUCKLE> - quiet laughter
    "⟨ꜱɪɢʜ⟩",       # <SIGH> - exhale sound
    "⟨ᴄᴏᴜɢʜ⟩",      # <COUGH> - cough sound
    "⟨ꜱɴɪꜰꜰʟᴇ⟩",   # <SNIFFLE> - nasal sound
    "⟨ɢʀᴏᴀɴ⟩",      # <GROAN> - vocalized exhale
    "⟨ʏᴀᴡɴ⟩",       # <YAWN> - yawning sound
    "⟨ɢᴀꜱᴘ⟩",       # <GASP> - sharp inhale
]


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + _special_vocals

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Special vocal token mappings for easy lookup
SPECIAL_VOCAL_MAP = {
    "<UH>": "⟨ᴜʜ⟩",
    "<UM>": "⟨ᴜᴍ⟩", 
    "<LAUGH>": "⟨ʟᴀᴜɢʜ⟩",
    "<GIGGLE>": "⟨ɢɪɢɢʟᴇ⟩",
    "<CHUCKLE>": "⟨ᴄʜᴜᴄᴋʟᴇ⟩",
    "<SIGH>": "⟨ꜱɪɢʜ⟩",
    "<COUGH>": "⟨ᴄᴏᴜɢʜ⟩",
    "<SNIFFLE>": "⟨ꜱɴɪꜰꜰʟᴇ⟩",
    "<GROAN>": "⟨ɢʀᴏᴀɴ⟩",
    "<YAWN>": "⟨ʏᴀᴡɴ⟩",
    "<GASP>": "⟨ɢᴀꜱᴘ⟩",
}
