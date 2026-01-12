from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
from markupsafe import Markup
import os
import traceback

from app.common.logger import get_logger

logger = get_logger(__name__)

from app.components.retriever import create_retriever_qa_chain

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)


def nl2br(value):
    if not isinstance(value, str):
        value = str(value)
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters["nl2br"] = nl2br



@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt", "").strip()

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})

            try:
                qa_chain = create_retriever_qa_chain()
                logger.info("Invoking QA chain.")
                response = qa_chain.invoke(user_input)
                logger.info("Successfully invoked QA chain.")
                messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                traceback.print_exc() 
                error_message = f"Error: An error occurred while processing your request: {str(e)}"
                messages.append({
                    "role": "assistant",
                    "content": error_message
                })

            session["messages"] = messages
            return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear_chat", methods=["GET"])
def clear_chat():
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
