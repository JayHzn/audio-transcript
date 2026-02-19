# app.py

import gradio as gr
from transcribe import pipeline

def launch_transcription(audio_file):
    """
    Gradio launch this function when clicking button.
    """
    
    if audio_file is None:
        return "No audio file upload.", None
    
    formatted_text, md_path = pipeline(audio_file)
    
    return formatted_text, md_path

# UI
with gr.Blocks(title="Audio Transcript -> Markdown") as app:
    
    gr.Markdown("# Transcript Audio to Markdown")
    gr.Markdown("Upload your audio file, click on **Transcript**, download your `.md` file for Notion.")
    
    audio_input = gr.Audio(
        label="Upload your audio file here",
        type="filepath",
    )
    
    button = gr.Button("Transcript", variant="primary")
    
    output_text = gr.Textbox(
        label="Results (Markdown)",
        lines=20,
    )
    
    output_file = gr.File(
        label="Download .md file"
    )
    
    button.click(
        fn=launch_transcription,
        inputs=[audio_input],
        outputs=[output_text, output_file]
    )
    
# Launching
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )