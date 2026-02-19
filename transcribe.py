# transcribe.py

from faster_whisper import WhisperModel
import ollama
import os
import shutil
from datetime import datetime

# Transcription
def transcript_audio(audio_path):
    """
    Take an audio file on input and return transcript text.
    
    - Model of Whisper : large-v3
    - Use Nvidia GPU : device="cuda"
    - compute_type="float16" to optimize speed on the GPU
    """
    
    audio_copy = os.path.join("output", "temp_audio" + os.path.splitext(audio_path)[1])
    shutil.copy2(audio_path, audio_copy)
    audio_path = audio_copy
    
    print("Loading of Whisper model...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    print("Transcript audio file...")
    segments, info = model.transcribe(audio_path, language="fr")
    
    print(f"Detected language : {info.language} (trust : {info.language_probability:.0%})")
    
    text = ""
    for segment in segments:
        text += segment.text + " "
    
    print(f"Transcription finished ({len(text.split())} words)")
    return text.strip()

def ollama_formatting(text):
    """
    Send text to Ollama to formatting to Markdown notes.
    """
    print("Ollama formatting")
    
    prompt = f"""Tu es un assistant qui transforme des transcriptions audio en notes structurées.

    Voici une transcription brute. Transforme-la en notes Markdown bien organisées avec :
    - Un titre principal (# Titre)
    - Les points clés organisés en sections (## Section)
    - Les éventuelles actions ou choses à retenir en fin de document

    Garde le sens exact du contenu, ne supprime aucune information importante.
    Corrige les erreurs de transcription évidentes.

    TRANSCRIPTION BRUTE :
    {text}
    """
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    
    formatted_text = response["message"]["content"]
    
    print("Formatting finished")
    return formatted_text

def save_md_file(formatted_text, output_dir="output"):
    """
    Save formatted text into .md file with a name based on datetime.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"transcription_{timestamp}.md"
    complete_path = os.path.join(output_dir, file_name)
    
    with open(complete_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    
    print(f"File saved : {complete_path}")
    return complete_path

def pipeline(audio_path):
    """
    Launch the entire pipeline :
        Audio -> Text -> Formatting -> .md file
    Gradio calls this function
    """
    
    text = transcript_audio(audio_path)
    formatting_text = ollama_formatting(text)
    md_path = save_md_file(formatting_text)
    
    return formatting_text, md_path