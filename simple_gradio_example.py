import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    gr.Markdown("## Basic Gradio App")
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Greeting")
    btn = gr.Button("Submit")

    btn.click(fn=greet, inputs=name, outputs=output)

demo.launch()
