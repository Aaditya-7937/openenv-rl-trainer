import gradio as gr

def greet():
    return 'Hello, OpenEnv RL Trainer is running. Check the logs or run main.py locally.'

if __name__ == '__main__':
    gr.Interface(fn=greet, inputs=None, outputs='text').launch()
