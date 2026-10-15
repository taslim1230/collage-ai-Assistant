import os
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def respond(message, history):
    if not HF_TOKEN:
        return "Please set HF_TOKEN in your .env file."

    messages = [
        {
            "role": "system",
            "content": (
                "You are College AI Assistant. Help students "
                "with admissions, fees, exams and scholarships. "
                "Reply in simple Hindi or English. Do not invent "
                "official college information."
            )
        }
    ]

    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({
                "role": "user",
                "content": user_msg
            })
        if bot_msg:
            messages.append({
                "role": "assistant",
                "content": bot_msg
            })

    messages.append({
        "role": "user",
        "content": message
    })

    try:
        client = InferenceClient(
            api_key=HF_TOKEN,
            provider="auto"
        )
        result = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=messages,
            max_tokens=500
        )
        return result.choices[0].message.content

    except Exception as e:
        return f"AI error: {e}"

demo = gr.ChatInterface(
    fn=respond,
    title="College AI Assistant 🎓",
    description="Ask questions about college, exams and scholarships."
)

if __name__ == "__main__":
    demo.launch()