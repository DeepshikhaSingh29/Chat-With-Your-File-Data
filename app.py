"""
Chat with Your Data
Upload a CSV/Excel file, ask questions in plain English.
The LLM writes pandas code, the app runs it, then the LLM explains the result.

Run:  streamlit run app.py
"""

import builtins
import io
import os
import re

import pandas as pd
import streamlit as st
from groq import Groq

# ---- Config -----------------------------------------------------------------
# Override the model without editing code:  set GROQ_MODEL=<model-name>
# See console.groq.com for the models currently available.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS = 2048          # reasoning models spend tokens "thinking", so keep this generous
MAX_RESULT_CHARS = 3000    # how much of the result we send back for explanation

CODE_SYSTEM_PROMPT = """You are a careful data analyst assistant.
You are given a pandas DataFrame named `df` and a question about it.
Write Python (pandas) code that answers the question.

Rules:
- Use only `df` and `pd`. Do NOT import anything. Do NOT read or write files.
- Store the final answer in a variable named `result`
  (a DataFrame, Series, number, or string).
- Return ONLY one ```python code block. No explanation.
- If the question cannot be answered from the columns available,
  set result = "Not answerable from this data: <short reason>".
"""

EXPLAIN_SYSTEM_PROMPT = """You are a data analyst explaining findings to a business user.
Given the question, the code that was run, and its result, reply in 2-4 plain-English
sentences. Mention specific numbers. Do not invent anything not in the result."""

# Builtins the generated code is allowed to use (no open, import, eval, etc.)
ALLOWED_BUILTINS = [
    "len", "range", "sum", "min", "max", "round", "sorted", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "abs", "enumerate", "zip",
    "any", "all", "reversed", "isinstance", "map", "filter", "pow", "divmod",
]
SAFE_BUILTINS = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}

# Quick static check on generated code (a guard, not a real sandbox).
BLOCKED_PATTERNS = ["import ", "__", "open(", "eval(", "exec(", "os.", "sys.", "subprocess"]


# ---- Helpers ----------------------------------------------------------------
def get_client() -> Groq:
    """Create the Groq client, or stop the app with a friendly message."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Set it as an environment variable, "
            "then restart the app. See the README for steps."
        )
        st.stop()
    return Groq(api_key=api_key)


def load_file(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def describe_df(df: pd.DataFrame) -> str:
    """Give the LLM the schema + a few sample rows (never the whole dataset)."""
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Column info:\n{buf.getvalue()}\n"
        f"First 5 rows:\n{df.head().to_string()}"
    )


def ask_llm(client: Groq, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,  # low randomness = more reliable code generation
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def extract_code(text: str) -> str:
    """Pull the code out of a ```python ... ``` block."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return match.group(1).strip() if match else text.strip()


def run_code(code: str, df: pd.DataFrame):
    """
    Execute the generated code with restricted builtins.
    NOTE: this is NOT a true sandbox. Fine for a local learning project;
    never expose it publicly without proper isolation.
    """
    if not code:
        raise ValueError("The model returned no code.")
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            raise ValueError(f"Generated code contains a blocked pattern: {pattern!r}")

    env = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd}
    exec(code, env)
    if "result" not in env:
        raise ValueError("The code did not define a variable named `result`.")
    return env["result"]


def result_to_text(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.to_string()
    else:
        text = str(result)
    return text[:MAX_RESULT_CHARS]


def answer_question(client: Groq, question: str, df: pd.DataFrame):
    """Generate code -> run it -> (retry once on error) -> explain."""
    schema = describe_df(df)
    prompt = f"DataFrame info:\n{schema}\n\nQuestion: {question}"

    code = extract_code(ask_llm(client, CODE_SYSTEM_PROMPT, prompt))
    try:
        result = run_code(code, df)
    except Exception as e:  # one self-repair attempt: show the LLM its own error
        retry_prompt = (
            f"{prompt}\n\nYour previous code:\n```python\n{code}\n```\n"
            f"It failed with: {type(e).__name__}: {e}\nFix it."
        )
        code = extract_code(ask_llm(client, CODE_SYSTEM_PROMPT, retry_prompt))
        result = run_code(code, df)  # if this fails too, the caller shows the error

    explanation = ask_llm(
        client,
        EXPLAIN_SYSTEM_PROMPT,
        f"Question: {question}\n\nCode:\n{code}\n\nResult:\n{result_to_text(result)}",
    )
    return code, result, explanation


# ---- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Chat with Your Data", page_icon="📊", layout="wide")
st.title("📊 Chat with Your Data")
st.caption("Upload a CSV or Excel file and ask questions in plain English.")

client = get_client()

uploaded = st.file_uploader("Upload your data", type=["csv", "xlsx"])

if uploaded:
    try:
        df = load_file(uploaded)
    except Exception as e:
        st.error(f"Could not read that file: {e}")
        st.stop()

    with st.expander(f"Preview ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
        st.dataframe(df.head(20))

    question = st.text_input(
        "Ask a question about your data",
        placeholder="e.g. What are the top 5 products by total revenue?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                code, result, explanation = answer_question(client, question, df)
            except Exception as e:
                st.error(f"Something went wrong: {type(e).__name__}: {e}")
            else:
                st.subheader("Answer")
                st.write(explanation)

                st.subheader("Result")
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    st.dataframe(result)
                else:
                    st.write(result)

                with st.expander("Show the code the AI wrote"):
                    st.code(code, language="python")
else:
    st.info("👆 Upload a file to get started.")
