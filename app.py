import gradio as gr
from agent import get_agent_response

MED_CSS = """
:root{
  --bg:#f6f8fb;
  --panel:#ffffff;
  --panel2:#ffffff;
  --text:#0f172a;
  --muted:#475569;
  --accent:#2563eb;   /* clinical blue */
  --warn:#f59e0b;
  --danger:#dc2626;
  --border:rgba(15,23,42,0.12);
  --radius:16px;
}

.gradio-container{
  max-width: 1100px !important;
  margin: 0 auto !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Header card */
#app_header{
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 14px 18px !important;
  margin-bottom: 14px !important;
  box-shadow: 0 8px 24px rgba(15,23,42,0.06) !important;
}

/* Chips */
.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: #f1f5f9;
  color: var(--muted);
  font-size: 12px;
  margin-left: 8px;
}

/* Callouts */
.callout{
  border: 1px solid var(--border);
  border-left: 4px solid var(--warn);
  background: #fff7ed;
  padding: 10px 12px;
  border-radius: 12px;
  color: var(--muted);
  margin-top: 10px;
}

.callout.danger{
  border-left-color: var(--danger);
  background: #fef2f2;
}

/* Chat surface */
#chatbot{
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--panel) !important;
  box-shadow: 0 8px 24px rgba(15,23,42,0.06) !important;
}

/* Buttons */
.gr-button-primary{
  background: var(--accent) !important;
  border: none !important;
  border-radius: 12px !important;
}

/* Inputs */
.gr-input, textarea{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: var(--panel) !important;
  color: var(--text) !important;
}

/* Small text */
.small-label{
  color: var(--muted);
  font-size: 12px;
}
"""

def create_gradio_interface():
    with gr.Blocks(title="Clinical Assistant", css=MED_CSS) as demo:
        gr.HTML("""
        <div id="app_header">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <div style="font-size:18px;font-weight:700;">
              🩺 Clinical Assistant <span class="badge">Evidence-informed</span><span class="badge">PHI-aware</span>
            </div>
            <div class="small-label">Version: 1.0</div>
          </div>
          <div class="callout">
            <b>Disclaimer:</b> Informational support only. Not a diagnosis or treatment plan.
          </div>
          <div class="callout danger">
            <b>Emergency:</b> If symptoms are severe (e.g., chest pain, trouble breathing, stroke signs), call 911 or local emergency services.
          </div>
        </div>
        """)

        with gr.Row(equal_height=True):
            # LEFT: context panel (optional)
            with gr.Column(scale=3):
                gr.Markdown("### Patient context (optional)")
                with gr.Row():
                    age = gr.Dropdown(
                        ["<18", "18–34", "35–49", "50–64", "65+"],
                        label="Age range",
                        value="18–34",
                    )
                    sex = gr.Dropdown(["Female", "Male", "Intersex", "Prefer not to say"], label="Sex", value="Prefer not to say")
                conditions = gr.Textbox(label="Known conditions", placeholder="e.g., diabetes, asthma", lines=2)
                meds = gr.Textbox(label="Current medications", placeholder="e.g., metformin 500mg daily", lines=2)
                allergies = gr.Textbox(label="Allergies", placeholder="e.g., penicillin", lines=1)

                gr.Markdown("### Quick prompts")
                with gr.Row():
                    p1 = gr.Button("Explain a diagnosis")
                    p2 = gr.Button("Medication side effects")
                    p3 = gr.Button("When to seek urgent care")
                with gr.Row():
                    p4 = gr.Button("Interpret labs (basic)")
                    p5 = gr.Button("Lifestyle recommendations")

            # RIGHT: chat
            with gr.Column(scale=7):
                history = gr.Chatbot(
                    elem_id="chatbot",
                    label="Chat",
                    show_label=False,
                    height=560,
                    type="messages",
                )

                msg = gr.Textbox(
                    label="Message",
                    placeholder="Describe symptoms, timeline, and what you’re trying to decide…",
                    show_label=False,
                    lines=2
                )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.ClearButton([msg, history], value="Clear")

                def build_context(age, sex, conditions, meds, allergies):
                    # Lightweight header added to every user message (optional)
                    parts = [
                        f"Age range: {age}",
                        f"Sex: {sex}",
                        f"Conditions: {conditions or 'N/A'}",
                        f"Medications: {meds or 'N/A'}",
                        f"Allergies: {allergies or 'N/A'}",
                    ]
                    return "Patient context:\n- " + "\n- ".join(parts)

                def user_submit(message, history, age, sex, conditions, meds, allergies):
                    if not message:
                        return "", history

                    ctx = build_context(age, sex, conditions, meds, allergies)
                    combined = f"{ctx}\n\nUser question:\n{message}"

                    history = history + [{"role": "user", "content": combined}]
                    return "", history

                async def call_agent(history):
                    if not history or history[-1]["role"] != "user":
                        return history

                    user_message, chat_history = history[-1]["content"], history[:-1]
                    response = await get_agent_response(user_message, chat_history)

                    # Encourage structured clinical output
                    if isinstance(response, str) and "Summary" not in response:
                        response = (
                            "## Summary\n" + response
                            + "\n\n## Next steps\n- If symptoms worsen, seek clinical care.\n"
                            + "\n## Red flags\n- Chest pain, difficulty breathing, fainting, severe headache, new weakness/numbness."
                        )

                    history.append({"role": "assistant", "content": response})
                    return history

                submit_btn.click(
                    user_submit,
                    inputs=[msg, history, age, sex, conditions, meds, allergies],
                    outputs=[msg, history],
                ).then(call_agent, inputs=history, outputs=history)

                # Quick prompt buttons
                def set_prompt(text):
                    return text

                p1.click(lambda: set_prompt("Explain this diagnosis in simple terms and what questions I should ask my clinician."), outputs=msg)
                p2.click(lambda: set_prompt("What are common side effects and serious warning signs for this medication?"), outputs=msg)
                p3.click(lambda: set_prompt("Based on these symptoms, what are red flags that need urgent care?"), outputs=msg)
                p4.click(lambda: set_prompt("Help interpret these lab results and what they could mean (non-diagnostic)."), outputs=msg)
                p5.click(lambda: set_prompt("What lifestyle changes are evidence-based for this condition?"), outputs=msg)

        return demo

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=8080, share=False)
