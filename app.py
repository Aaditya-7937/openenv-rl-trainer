import gradio as gr
import subprocess
import os

def run_training():
    try:
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
        logs = result.stdout
        if result.stderr:
            logs += '
Errors/Warnings:
' + result.stderr
        
        img_path = './results/training_results.png'
        if os.path.exists(img_path):
            return logs, img_path
        else:
            return logs, None
    except Exception as e:
        return f'Error: {str(e)}', None

with gr.Blocks() as demo:
    gr.Markdown('# OpenEnv RL Trainer')
    gr.Markdown('Target Environment: **https://aaditya-7937-openenv-review.hf.space**')
    
    with gr.Row():
        start_btn = gr.Button('Start Training Pipeline', variant='primary')
    
    with gr.Row():
        with gr.Column():
            log_output = gr.Textbox(label='Training Logs', lines=25)
        with gr.Column():
            plot_output = gr.Image(label='Training Results Plot', type='filepath')
            
    start_btn.click(fn=run_training, inputs=None, outputs=[log_output, plot_output])

if __name__ == '__main__':
    demo.launch()
