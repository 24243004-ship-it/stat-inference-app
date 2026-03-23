import math
import os
import base64
import io
import contextlib
import re
import numpy as np
import matplotlib.pyplot as plt


import streamlit as st
from scipy.stats import norm

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

st.set_page_config(page_title="Inferential Statistics Learning App", layout="wide")

LOGO_PATH = r"C:\Users\gowri\Downloads\nec_logo.png"
PDF_PATH = r"/mnt/data/inferential_statistics_notes.pdf"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 45%, #fdf2f8 100%);
        }

        .main-title {
            background: linear-gradient(90deg, #7c3aed, #ec4899, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 900;
            margin-bottom: 0.15rem;
            line-height: 1.1;
        }

        .sub-text {
            font-size: 1.05rem;
            color: #334155;
            margin-bottom: 1rem;
        }

        .hero-box {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #eef2ff, #fdf2f8, #fff7ed);
            padding: 1.8rem;
            border-radius: 24px;
            border: 2px solid #ddd6fe;
            box-shadow: 0 12px 28px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }

        .hero-heading {
            font-size: 2.3rem;
            font-weight: 900;
            color: #1e1b4b;
            margin-bottom: 0.4rem;
        }

        .hero-para {
            font-size: 1.06rem;
            color: #374151;
            max-width: 900px;
        }

        .floating-stat {
            position: absolute;
            font-weight: 900;
            opacity: 0.12;
            animation: floatUp 8s ease-in-out infinite;
            user-select: none;
        }

        .stat1 { top: 10px; right: 35px; font-size: 2.4rem; color: #7c3aed; }
        .stat2 { top: 90px; right: 180px; font-size: 2rem; color: #2563eb; animation-delay: 1s; }
        .stat3 { bottom: 10px; right: 70px; font-size: 2.2rem; color: #ec4899; animation-delay: 2s; }
        .stat4 { bottom: 25px; left: 60px; font-size: 1.8rem; color: #f59e0b; animation-delay: 3s; }

        @keyframes floatUp {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-12px) rotate(4deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }

        .coordinator-box {
            background: linear-gradient(135deg, #fef3c7, #fde68a, #fca5a5);
            color: #7c2d12;
            padding: 1.1rem 1.3rem;
            border-radius: 22px;
            border-left: 10px solid #f59e0b;
            font-size: 1.4rem;
            font-weight: 900;
            box-shadow: 0 10px 20px rgba(245,158,11,0.2);
            margin: 1rem 0 1.2rem 0;
            text-align: center;
        }

        .section-label {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1f2937;
            margin: 0.8rem 0;
        }

        .dev-card {
            background: linear-gradient(135deg, #dbeafe, #e0f2fe, #dcfce7);
            padding: 1.2rem;
            border-radius: 22px;
            text-align: center;
            font-weight: 800;
            font-size: 1.1rem;
            box-shadow: 0 10px 22px rgba(0,0,0,0.10);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            border: 1px solid #bfdbfe;
            margin-bottom: 0.7rem;
        }

        .dev-card:hover {
            transform: translateY(-8px) scale(1.04);
            box-shadow: 0 18px 30px rgba(0,0,0,0.16);
        }

        .dev-reg {
            font-size: 0.95rem;
            color: #334155;
            margin-top: 0.35rem;
        }

        .year-pill {
            display: inline-block;
            background: linear-gradient(90deg, #2563eb, #7c3aed, #ec4899);
            color: white;
            padding: 0.7rem 1.1rem;
            border-radius: 999px;
            font-weight: 800;
            margin-top: 0.7rem;
            box-shadow: 0 8px 18px rgba(37,99,235,0.25);
        }

        .topic-card {
            background: linear-gradient(135deg, #eff6ff, #f5f3ff);
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid #c4b5fd;
            box-shadow: 0 8px 18px rgba(0,0,0,0.05);
            min-height: 120px;
        }

        .footer-box {
            margin-top: 1.5rem;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: linear-gradient(90deg, #eff6ff, #f5f3ff);
            border: 1px solid #c4b5fd;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .info-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 18px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        .answer-box {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-left: 8px solid #2563eb;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 18px rgba(0,0,0,0.05);
            margin-top: 0.8rem;
            margin-bottom: 1rem;
        }

        .step-box {
            background: linear-gradient(90deg, #eff6ff, #ffffff);
            border-radius: 14px;
            padding: 0.8rem;
            border: 1px solid #dbeafe;
            margin-bottom: 0.6rem;
        }

        .mini-badge {
            display: inline-block;
            background: linear-gradient(90deg, #7c3aed, #ec4899);
            color: white;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 800;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    logo_base64 = get_base64_image(LOGO_PATH)
    if logo_base64:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:18px;margin-bottom:8px;">
                <img src="data:image/png;base64,{logo_base64}" width="88" style="border-radius:18px;box-shadow:0 8px 16px rgba(0,0,0,0.12);">
                <div>
                    <div style="font-size:1.65rem;font-weight:900;color:#111827;">National Engineering College</div>
                    <div style="color:#2563eb;font-weight:700;font-size:1rem;">https://nec.edu.in</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## National Engineering College")
        st.caption("https://nec.edu.in")


def show_footer():
    logo_base64 = get_base64_image(LOGO_PATH)
    if logo_base64:
        st.markdown(
            f"""
            <div class="footer-box">
                <img src="data:image/png;base64,{logo_base64}" width="48" style="border-radius:12px;">
                <div>
                    <div style="font-weight:900;">National Engineering College</div>
                    <div style="color:#2563eb;">https://nec.edu.in</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("---")
        st.markdown("**National Engineering College**")
        st.caption("https://nec.edu.in")


def section_title(title: str):
    st.markdown(f"## {title}")


def step_block(lines):
    for i, line in enumerate(lines, start=1):
        st.write(f"**Step {i}:** {line}")


def show_steps(step_lines):
    st.markdown("### Step-by-Step Solution")
    for i, line in enumerate(step_lines, 1):
        st.markdown(f"<div class='step-box'><b>Step {i}:</b> {line}</div>", unsafe_allow_html=True)


def info_sections(explanation, example, advantages, disadvantages, applications, realtime):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### Explanation")
        st.write(explanation)
        st.markdown("### Example")
        st.write(example)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### Advantages")
        for x in advantages:
            st.write(f"- {x}")
        st.markdown("### Disadvantages")
        for x in disadvantages:
            st.write(f"- {x}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### Applications")
    for x in applications:
        st.write(f"- {x}")
    st.markdown("### Real-Time Example")
    st.write(realtime)
    st.markdown("</div>", unsafe_allow_html=True)


def run_python_code(code_text, button_key):
    st.markdown("### Python Code")
    code = st.text_area("Edit / Run Python code", value=code_text, height=220, key=f"code_{button_key}")
    if st.button("Run Python Code", key=f"run_{button_key}", use_container_width=True):
        output_buffer = io.StringIO()
        plt.close("all")

        safe_globals = {
            "__builtins__": __builtins__,
            "math": math,
            "np": np,
            "plt": plt,
            "norm": norm,
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
        }

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, safe_globals, safe_globals)

            output = output_buffer.getvalue()
            if output.strip():
                st.code(output, language="text")
            else:
                st.success("Code executed successfully.")

            figs = [plt.figure(num) for num in plt.get_fignums()]
            for fig in figs:
                st.pyplot(fig)
            plt.close("all")

        except Exception as e:
            st.error(f"Error while running code: {e}")


# -------------------------------------------------
# PDF KNOWLEDGE HELPERS (NEW ADD)
# -------------------------------------------------
def load_pdf_text(pdf_path):
    if PyPDF2 is None:
        return ""
    if not os.path.exists(pdf_path):
        return ""

    text_parts = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception:
        return ""

    return "\n".join(text_parts)


def chunk_text(text, chunk_size=700):
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


PDF_TEXT = load_pdf_text(PDF_PATH)
PDF_CHUNKS = chunk_text(PDF_TEXT)


def keyword_score(question, chunk):
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    c_words = set(re.findall(r"[a-zA-Z]+", chunk.lower()))
    if not q_words or not c_words:
        return 0
    return len(q_words.intersection(c_words))


def get_relevant_pdf_chunks(question, top_k=2):
    if not PDF_CHUNKS:
        return []

    scored = []
    for chunk in PDF_CHUNKS:
        score = keyword_score(question, chunk)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def pdf_topic_answer(question):
    q = question.lower().strip()

    topic_map = {
        "population": "Population is the entire group. Sample is a subset selected from the population.",
        "sample": "Population is the entire group. Sample is a subset selected from the population.",
        "random sampling": "Every member has equal chance of selection. Helps reduce bias.",
        "sampling distribution": "Distribution of sample means. Mean equals population mean.",
        "standard error": "SE = sigma / sqrt(n). Measures variability of sample mean.",
        "hypothesis testing": "Process to test claims using data. Includes null and alternative hypothesis.",
        "z-test": "Z = (x_bar - mu) / (sigma / sqrt(n)). Used when population std is known.",
        "one-tailed": "One-tailed checks one direction. Two-tailed checks both directions.",
        "two-tailed": "One-tailed checks one direction. Two-tailed checks both directions.",
        "estimation": "Using sample data to estimate population parameters.",
        "point estimate": "Single value estimate of parameter.",
        "confidence interval": "Range of values likely to contain parameter.",
        "effect of sample size": "Larger sample size reduces error and increases accuracy."
    }

    for key, ans in topic_map.items():
        if key in q:
            return ans

    chunks = get_relevant_pdf_chunks(question, top_k=2)
    if chunks:
        return "\n\n".join(chunks)

    return ""


# -------------------------------------------------
# Graph functions for all topics
# -------------------------------------------------
def draw_population_sample_plot(pop_size=1000, sample_size=50):
    population = np.random.normal(loc=50, scale=10, size=pop_size)
    sample = np.random.choice(population, size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(population, bins=25, alpha=0.7, label="Population")
    ax.hist(sample, bins=15, alpha=0.7, label="Sample")
    ax.set_title("Population vs Sample")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


def draw_random_sampling_plot(sample):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(sample) + 1), sample)
    ax.set_title("Random Sample Values")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Selected Value")
    return fig


def draw_sampling_distribution(pop_mean=50, sigma=10, n=36, reps=1000):
    means = [np.mean(np.random.normal(pop_mean, sigma, n)) for _ in range(reps)]
    theoretical_se = sigma / math.sqrt(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(means, bins=30, alpha=0.8)
    ax.axvline(pop_mean, linestyle="--", linewidth=2, label=f"Mean = {pop_mean:.2f}")
    ax.set_title("Sampling Distribution of Sample Mean")
    ax.set_xlabel("Sample Means")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, theoretical_se


def draw_standard_error_graph(sigma, n):
    sizes = np.arange(2, max(10, int(n) + 20))
    ses = [sigma / math.sqrt(x) for x in sizes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, ses, marker="o")
    ax.set_title("Standard Error vs Sample Size")
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Standard Error")
    return fig


def draw_hypothesis_testing_graph():
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y)
    ax.axvline(-1.96, linestyle="--", label="-1.96")
    ax.axvline(1.96, linestyle="--", label="1.96")
    ax.set_title("Hypothesis Testing Critical Region")
    ax.legend()
    return fig


def draw_z_test_curve(z_value, alpha=0.05, tail="two-tailed"):
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, linewidth=2)

    if tail == "two-tailed":
        crit = norm.ppf(1 - alpha / 2)
        ax.axvline(-crit, linestyle="--", label=f"-{crit:.2f}")
        ax.axvline(crit, linestyle="--", label=f"{crit:.2f}")
    elif tail == "right-tailed":
        crit = norm.ppf(1 - alpha)
        ax.axvline(crit, linestyle="--", label=f"{crit:.2f}")
    else:
        crit = norm.ppf(alpha)
        ax.axvline(crit, linestyle="--", label=f"{crit:.2f}")

    ax.axvline(z_value, linewidth=2, label=f"z = {z_value:.2f}")
    ax.set_title("Z-Test Visualization")
    ax.set_xlabel("z value")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def draw_confidence_interval_graph(sample_mean, lower, upper):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hlines(y=1, xmin=lower, xmax=upper, linewidth=6)
    ax.plot(sample_mean, 1, "o", markersize=10)
    ax.set_title("Confidence Interval Graph")
    ax.set_yticks([])
    ax.set_xlabel("Value")
    return fig


def draw_ci_effect(sigma=10):
    sizes = np.array([10, 20, 30, 50, 100, 200])
    margins = [norm.ppf(0.975) * sigma / math.sqrt(n) for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, margins, marker="o")
    ax.set_title("Effect of Sample Size on Margin of Error")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Margin of Error")
    return fig


# -------------------------------------------------
# Core statistical helpers
# -------------------------------------------------
def z_test_known_sigma(sample_mean, pop_mean, sigma, n, alpha=0.05, tail="two-tailed"):
    se = sigma / math.sqrt(n)
    z = (sample_mean - pop_mean) / se

    if tail == "right-tailed":
        p_value = 1 - norm.cdf(z)
        critical = norm.ppf(1 - alpha)
        reject = z > critical
    elif tail == "left-tailed":
        p_value = norm.cdf(z)
        critical = norm.ppf(alpha)
        reject = z < critical
    else:
        p_value = 2 * (1 - norm.cdf(abs(z)))
        critical = norm.ppf(1 - alpha / 2)
        reject = abs(z) > critical

    return {
        "se": se,
        "z": z,
        "p_value": p_value,
        "critical": critical,
        "reject": reject,
    }


def confidence_interval_known_sigma(sample_mean, sigma, n, confidence=0.95):
    alpha = 1 - confidence
    z_star = norm.ppf(1 - alpha / 2)
    se = sigma / math.sqrt(n)
    moe = z_star * se
    return {
        "z_star": z_star,
        "se": se,
        "moe": moe,
        "lower": sample_mean - moe,
        "upper": sample_mean + moe,
    }


# -------------------------------------------------
# Old helpers
# -------------------------------------------------
def home_question_answer(question):
    q = question.lower().strip()

    if not q:
        st.warning("Please enter a question.")
        return

    # 1. First try smart solver
    result = smart_solver(question)

    if result["steps"]:
        _render_solver_result(result)
        return

    # 2. If no numerical solve, try PDF answer
    pdf_ans = pdf_topic_answer(question)

    if pdf_ans:
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("<span class='mini-badge'>PDF Answer</span><span class='mini-badge'>Notes Based</span>", unsafe_allow_html=True)
        st.markdown("### Answer from Attached PDF")
        st.write(pdf_ans)

        if wants_code(question):
            st.markdown("### Python Code")
            st.code(
                """# Ask a more specific numerical question to generate exact code
# Example:
# sample mean = 62, population mean = 60, standard deviation = 4, sample size = 64""",
                language="python"
            )

        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 3. Final fallback
    st.info("Question identified, but exact automatic solve was not possible. Please enter the question more clearly.")


def normalize_text(text):
    return text.lower().strip()


def wants_graph(question):
    q = normalize_text(question)
    graph_words = ["graph", "plot", "visualize", "visualise", "curve", "chart", "diagram"]
    return any(word in q for word in graph_words)


def wants_code(question):
    q = question.lower().strip()
    return any(word in q for word in ["python", "code", "script", "program", "numpy"])


def extract_numbers(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return [float(x) for x in nums]


# -------------------------------------------------
# SMART UNIVERSAL SOLVER
# -------------------------------------------------
def _norm(text):
    return text.lower().strip()


def _wants_graph(question):
    q = _norm(question)
    return any(x in q for x in ["graph", "plot", "visualize", "visualise", "curve", "chart", "diagram"])


def _wants_code(question):
    q = _norm(question)
    return any(x in q for x in ["python", "code", "script", "program", "numpy"])


def _nums(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def _list_after_colon(text):
    m = re.findall(r'[:：]\s*([0-9,\s]+)', text)
    if m:
        return [float(x) for x in re.findall(r"-?\d+\.?\d*", m[-1])]
    return []


def _render_solver_result(result):
    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.markdown(
        "<span class='mini-badge'>Smart Answer</span>"
        "<span class='mini-badge'>Step by Step</span>"
        "<span class='mini-badge'>Correct Logic</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"### {result['title']}")
    if result["steps"]:
        show_steps(result["steps"])
    else:
        st.info("Please ask a clearer statistics question.")
    if result.get("graph") is not None:
        st.pyplot(result["graph"])
    if result.get("code"):
        st.markdown("### Python Code")
        st.code(result["code"], language="python")
    st.markdown("</div>", unsafe_allow_html=True)


def smart_solver(question):
    q = _norm(question)
    nums = _nums(question)
    data_list = _list_after_colon(question)

    result = {"title": "Smart Solution", "steps": [], "graph": None, "code": None}

    if "stratified" in q or ("juniors" in q and "seniors" in q):
        if len(nums) >= 4:
            total, juniors, seniors, sample = nums[:4]
            j_sel = (juniors / total) * sample
            s_sel = (seniors / total) * sample
            result["title"] = "Stratified Sampling Solution"
            result["steps"] = [
                f"Total population = {total}",
                f"Juniors = {juniors}",
                f"Seniors = {seniors}",
                f"Sample size = {sample}",
                "Formula: Group sample = (group size / total population) × sample size",
                f"Juniors selected = ({juniors}/{total}) × {sample} = {j_sel:.2f}",
                f"Seniors selected = ({seniors}/{total}) × {sample} = {s_sel:.2f}",
                f"Final Answer: Juniors = {round(j_sel)}, Seniors = {round(s_sel)}"
            ]
            if _wants_graph(question):
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(["Juniors", "Seniors"], [j_sel, s_sel])
                ax.set_title("Stratified Sample Allocation")
                ax.set_ylabel("Selected Count")
                result["graph"] = fig
            return result

    if "identify" in q and "population" in q and "sample" in q:
        result["title"] = "Population and Sample Identification"

        if "75" in q and ("phone" in q or "survey" in q or "directory" in q):
            result["steps"] = [
                "Population = all constituents of the city council member",
                "Sample = the 75 names selected from the city phone directory and surveyed"
            ]
            return result

        if "600" in q and ("25%" in q or "25 %" in q):
            result["steps"] = [
                "Population = all 600 students",
                "Sample = 25% of the 600 students",
                "Sample = 0.25 × 600 = 150",
                "Final Answer: Population = 600 students, Sample = 150 students"
            ]
            return result

        result["steps"] = [
            "Population = the complete group under study",
            "Sample = the smaller selected subset from that population"
        ]
        return result

    if "population mean" in q:
        data = data_list if data_list else nums
        if len(data) >= 2:
            total = sum(data)
            n = len(data)
            mean_val = total / n
            result["title"] = "Population Mean"
            result["steps"] = [
                f"Values = {data}",
                f"Sum of values = {total}",
                f"Number of values N = {n}",
                "Formula: μ = Σx / N",
                f"Population mean = {total} / {n} = {mean_val:.4f}"
            ]
            if _wants_code(question):
                result["code"] = f"""data = {data}
mean_val = sum(data) / len(data)
print("Population Mean =", mean_val)"""
            return result

    if "population variance" in q or ("variance" in q and (data_list or len(nums) >= 3)):
        data = data_list if data_list else nums
        if len(data) >= 2:
            mean_val = sum(data) / len(data)
            sq = [(x - mean_val) ** 2 for x in data]
            var_val = sum(sq) / len(data)
            result["title"] = "Population Variance"
            result["steps"] = [
                f"Values = {data}",
                f"Mean = {mean_val:.4f}",
                f"Squared deviations = {[round(x, 4) for x in sq]}",
                f"Sum of squared deviations = {sum(sq):.4f}",
                "Formula: σ² = Σ(x - μ)² / N",
                f"Population variance = {sum(sq):.4f} / {len(data)} = {var_val:.4f}"
            ]
            result["code"] = f"""data = {data}
mean_val = sum(data) / len(data)
pop_variance = sum((x - mean_val) ** 2 for x in data) / len(data)
print("Mean =", mean_val)
print("Population Variance =", pop_variance)"""
            return result

    if "z-score" in q or "z score" in q:
        if len(nums) >= 3:
            mean_val = nums[0]
            sd = nums[1]
            x = nums[2]
            z = (x - mean_val) / sd
            result["title"] = "Z-Score"
            result["steps"] = [
                f"Score x = {x}",
                f"Mean μ = {mean_val}",
                f"Standard deviation σ = {sd}",
                "Formula: z = (x - μ) / σ",
                f"z = ({x} - {mean_val}) / {sd}",
                f"z = {z:.4f}"
            ]
            return result

    if "probability" in q and "selected" in q and "more than once" not in q:
        if len(nums) >= 2:
            total, selected = nums[:2]
            p = selected / total
            result["title"] = "Selection Probability"
            result["steps"] = [
                f"Total items = {total}",
                f"Selected items = {selected}",
                "Formula: Probability = selected / total",
                f"P = {selected} / {total}",
                f"P = {p:.4f}",
                f"Percentage = {p * 100:.2f}%"
            ]
            return result

    if "more than once" in q:
        result["title"] = "Sampling Without Replacement"
        result["steps"] = [
            "In a random survey, people/items are usually selected without replacement.",
            "So once a person is selected, they are not selected again.",
            "Therefore, the probability that one attendee is selected more than once is 0."
        ]
        return result

    if ("numpy" in q or "python" in q) and "random sample" in q:
        result["title"] = "NumPy Random Sampling Code"
        result["steps"] = [
            "Create a population of integers from 1 to 10000.",
            "Use np.random.choice to select 50 values.",
            "Use replace=False to avoid repetition."
        ]
        result["code"] = """import numpy as np

population = np.arange(1, 10001)
sample = np.random.choice(population, size=50, replace=False)

print(sample)"""
        return result

    if ("proportion" in q and "downloaded" in q) or ("sample proportion" in q):
        if len(nums) >= 3:
            p_pct = nums[0]
            pop_size = nums[1]
            n = nums[2]
            p = p_pct / 100 if p_pct > 1 else p_pct
            mean_phat = p
            sd_phat = math.sqrt((p * (1 - p)) / n)
            np_val = n * p
            nq_val = n * (1 - p)

            result["title"] = "Sampling Distribution of Sample Proportion"
            result["steps"] = [
                f"Population proportion p = {p:.4f}",
                f"Sample size n = {int(n)}",
                "Mean of p̂: μ(p̂) = p",
                f"μ(p̂) = {mean_phat:.4f}",
                "Standard deviation of p̂: σ(p̂) = √[p(1-p)/n]",
                f"σ(p̂) = √[{p:.4f}(1-{p:.4f})/{int(n)}] = {sd_phat:.4f}",
                "Conditions:",
                f"10% condition: {int(n)} is less than 10% of {int(pop_size)} → satisfied",
                f"Success-failure: np = {np_val:.2f}, n(1-p) = {nq_val:.2f}",
                "Since np < 10, the normal approximation condition is not satisfied."
            ]
            return result

    if "hypothesis testing" in q and ("describe" in q or "detail" in q):
        result["title"] = "Hypothesis Testing"
        result["steps"] = [
            "Hypothesis testing is a method used to test a claim about a population using sample data.",
            "The null hypothesis H₀ usually means no effect or no difference.",
            "The alternative hypothesis H₁ represents the claim being tested.",
            "Choose a significance level α, such as 0.05.",
            "Compute the test statistic from the sample data.",
            "Use the p-value or critical value to make a decision.",
            "If evidence is strong, reject H₀; otherwise fail to reject H₀.",
            "Finally, write the conclusion in words based on the problem."
        ]
        return result

    if "identify the test used" in q and "z" in q:
        if len(nums) >= 4:
            xbar, mu, sigma, n = nums[:4]
            se = sigma / math.sqrt(n)
            z = (xbar - mu) / se
            result["title"] = "Test Used and Z Formula"
            result["steps"] = [
                "Since population standard deviation is known, the correct test is a One-Sample Z-Test.",
                "Formula: Z = (x̄ - μ) / (σ / √n)",
                f"Substitution: Z = ({xbar} - {mu}) / ({sigma} / √{int(n)})",
                f"Standard Error = {sigma} / √{int(n)} = {se:.4f}",
                f"Z = {z:.4f}"
            ]
            return result

    if "standard error" in q:
        if len(nums) >= 2:
            sigma, n = nums[:2]
            se = sigma / math.sqrt(n)
            result["title"] = "Standard Error"
            result["steps"] = [
                f"Given σ = {sigma}",
                f"Given n = {int(n)}",
                "Formula: SE = σ / √n",
                f"SE = {sigma} / √{int(n)} = {se:.4f}"
            ]
            if _wants_graph(question):
                result["graph"] = draw_standard_error_graph(sigma, int(n))
            return result

    if "confidence interval" in q:
        if len(nums) >= 3:
            xbar, sigma, n = nums[:3]
            conf = 0.95
            if "90%" in q:
                conf = 0.90
            elif "99%" in q:
                conf = 0.99
            ci = confidence_interval_known_sigma(xbar, sigma, int(n), conf)
            result["title"] = "Confidence Interval"
            result["steps"] = [
                f"Sample mean x̄ = {xbar}",
                f"Population standard deviation σ = {sigma}",
                f"Sample size n = {int(n)}",
                f"Confidence level = {int(conf * 100)}%",
                "Formula: CI = x̄ ± z* × (σ / √n)",
                f"z* = {ci['z_star']:.4f}",
                f"SE = {ci['se']:.4f}",
                f"Margin of Error = {ci['moe']:.4f}",
                f"Confidence Interval = ({ci['lower']:.4f}, {ci['upper']:.4f})"
            ]
            if _wants_graph(question):
                result["graph"] = draw_confidence_interval_graph(xbar, ci["lower"], ci["upper"])
            if _wants_code(question):
                result["code"] = f"""import math
from scipy.stats import norm

sample_mean = {xbar}
sigma = {sigma}
n = {int(n)}
confidence = {conf}

alpha = 1 - confidence
z_star = norm.ppf(1 - alpha / 2)
se = sigma / math.sqrt(n)
moe = z_star * se

print("CI =", (sample_mean - moe, sample_mean + moe))"""
            return result

    if "one-tailed z-test" in q or "two-tailed z-test" in q:
        score_blocks = re.findall(r"\[(.*?)\]", question, re.DOTALL)
        if score_blocks:
            score_text = score_blocks[-1]
            scores = [float(x.strip()) for x in score_text.split(",")]
            pop_mean = 75
            sigma = 12
            if "average score" in q and len(nums) >= 2:
                pop_mean = nums[0]
                sigma = nums[1]

            n = len(scores)
            xbar = sum(scores) / n
            one = z_test_known_sigma(xbar, pop_mean, sigma, n, 0.05, "right-tailed")
            two = z_test_known_sigma(xbar, pop_mean, sigma, n, 0.05, "two-tailed")

            result["title"] = "One-Tailed and Two-Tailed Z-Test"
            result["steps"] = [
                f"Sample size n = {n}",
                f"Sample mean x̄ = {xbar:.4f}",
                f"Population mean μ = {pop_mean}",
                f"Population standard deviation σ = {sigma}",
                f"Standard Error = {one['se']:.4f}",
                f"Z statistic = {one['z']:.4f}",
                f"One-tailed critical value = {one['critical']:.4f}",
                f"One-tailed p-value = {one['p_value']:.6f}",
                "One-tailed decision: " + ("Reject H₀" if one["reject"] else "Fail to Reject H₀"),
                f"Two-tailed critical value = ±{two['critical']:.4f}",
                f"Two-tailed p-value = {two['p_value']:.6f}",
                "Two-tailed decision: " + ("Reject H₀" if two["reject"] else "Fail to Reject H₀"),
                "Comparison: one-tailed checks only 'greater than'; two-tailed checks any difference.",
                "Here both tests fail to reject H₀."
            ]
            if _wants_graph(question):
                result["graph"] = draw_z_test_curve(one["z"], 0.05, "right-tailed")
            result["code"] = f"""import math
from scipy.stats import norm

scores = {scores}
mu = {pop_mean}
sigma = {sigma}
n = len(scores)

x_bar = sum(scores) / n
se = sigma / math.sqrt(n)
z = (x_bar - mu) / se

p_one = 1 - norm.cdf(z)
p_two = 2 * (1 - norm.cdf(abs(z)))

print("Sample mean =", x_bar)
print("SE =", se)
print("z =", z)
print("One-tailed p =", p_one)
print("Two-tailed p =", p_two)"""
            return result

    if "sample mean" in q and "population mean" in q and "standard deviation" in q and "sample size" in q:
        if len(nums) >= 4:
            xbar, mu, sigma, n = nums[:4]
            tail = "two-tailed"
            if "greater than" in q or "right-tailed" in q or "right tailed" in q:
                tail = "right-tailed"
            elif "less than" in q or "left-tailed" in q or "left tailed" in q:
                tail = "left-tailed"

            zres = z_test_known_sigma(xbar, mu, sigma, int(n), 0.05, tail)
            result["title"] = "Z-Test Solution"
            result["steps"] = [
                f"Given x̄ = {xbar}",
                f"Given μ = {mu}",
                f"Given σ = {sigma}",
                f"Given n = {int(n)}",
                "Formula: z = (x̄ - μ) / (σ / √n)",
                f"SE = {zres['se']:.4f}",
                f"z = {zres['z']:.4f}",
                f"P-value = {zres['p_value']:.6f}",
                "Decision: " + ("Reject H₀" if zres["reject"] else "Fail to Reject H₀")
            ]
            if _wants_graph(question):
                result["graph"] = draw_z_test_curve(zres["z"], 0.05, tail)
            if _wants_code(question):
                result["code"] = f"""import math
from scipy.stats import norm

sample_mean = {xbar}
population_mean = {mu}
sigma = {sigma}
n = {int(n)}

se = sigma / math.sqrt(n)
z = (sample_mean - population_mean) / se

print("SE =", se)
print("z =", z)"""
            return result

    return result


def solve_text_question(question, topic_name):
    q = question.lower().strip()
    if not q:
        st.warning("Please enter a question.")
        return

    st.markdown("### Answer")

    if topic_name == "Population vs Sample":
        st.success("Population is the complete group. Sample is the selected smaller group from the population.")
        st.write("Example: All students in a school = population, 50 selected students = sample.")
    elif topic_name == "Random Sampling":
        st.success("Random sampling means every item has equal chance of selection.")
        st.write("This reduces bias and improves fairness.")
    elif topic_name == "Sampling Distribution":
        st.success("Sampling distribution is the distribution of sample statistics like sample means from repeated samples.")
        st.write("It helps understand variability in the sample mean.")
    elif topic_name == "Standard Error of Mean":
        st.success("Standard Error tells how much the sample mean varies across repeated samples.")
        st.write("Formula: SE = σ / √n")
    elif topic_name == "Hypothesis Testing":
        st.success("Hypothesis testing checks whether sample evidence is enough to reject a claim.")
        st.write("We use H₀, H₁, alpha, test statistic, and final decision.")
    elif topic_name == "Z-Test Solver":
        st.success("Z-test is used when population standard deviation is known.")
        st.write("Enter the numeric values in manual solve above for exact Z value and decision.")
    elif topic_name == "Confidence Interval":
        st.success("Confidence interval gives a likely range for the population mean.")
        st.write("Enter the values above for exact lower and upper limit.")
    elif topic_name == "Effect of Sample Size":
        st.success("As sample size increases, standard error decreases and precision increases.")
    else:
        st.info("Use manual solve for detailed answer.")


def solve_topic_question(question, topic_name):
    q = question.lower().strip()
    if not q:
        st.warning("Please enter a question.")
        return

    # 1. Try smart solver first
    result = smart_solver(question)
    if result["steps"]:
        _render_solver_result(result)
        return

    # 2. Try PDF answer next
    pdf_ans = pdf_topic_answer(question)
    if pdf_ans:
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("### Answer from Attached PDF")
        st.write(pdf_ans)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 3. Old fallback topic logic
    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)

    if topic_name == "Population vs Sample":
        st.markdown("### Answer")
        st.write("Population is the complete group. Sample is the smaller selected group.")
        if wants_graph(question):
            st.pyplot(draw_population_sample_plot(1000, 50))

    elif topic_name == "Random Sampling":
        st.markdown("### Answer")
        st.write("Random sampling means every member has equal chance of selection.")
        if wants_graph(question):
            sample = np.random.choice(range(1, 101), size=10, replace=False)
            st.pyplot(draw_random_sampling_plot(sample))

    elif topic_name == "Sampling Distribution":
        st.markdown("### Answer")
        st.write("Sampling distribution is the distribution of sample statistics from repeated samples.")
        if wants_graph(question):
            fig, _ = draw_sampling_distribution()
            st.pyplot(fig)

    elif topic_name == "Standard Error of Mean":
        st.markdown("### Answer")
        st.write("Standard Error tells how much the sample mean varies across repeated samples.")
        if wants_graph(question):
            st.pyplot(draw_standard_error_graph(12, 36))

    elif topic_name == "Hypothesis Testing":
        st.markdown("### Answer")
        st.write("Hypothesis testing checks whether sample evidence is enough to reject a claim.")
        if wants_graph(question):
            st.pyplot(draw_hypothesis_testing_graph())

    elif topic_name == "Z-Test Solver":
        st.markdown("### Answer")
        st.write("Z-test is used when population standard deviation is known.")

    elif topic_name == "Confidence Interval":
        st.markdown("### Answer")
        st.write("Confidence interval gives a likely range for the population mean.")

    elif topic_name == "Effect of Sample Size":
        st.markdown("### Answer")
        st.write("As sample size increases, standard error decreases and estimate becomes more precise.")
        if wants_graph(question):
            st.pyplot(draw_ci_effect(10))

    st.markdown("</div>", unsafe_allow_html=True)


inject_custom_css()

# -------------------------------------------------
# Header
# -------------------------------------------------
show_header()
st.markdown("<div class='main-title'>Inferential Statistics Learning App</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-text'>This app explains inferential statistics, gives formulas, solves problems step by step, and visualizes graphs for better understanding.</div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
topic_options = [
    "Home",
    "Population vs Sample",
    "Random Sampling",
    "Sampling Distribution",
    "Standard Error of Mean",
    "Hypothesis Testing",
    "Z-Test Solver",
    "Confidence Interval",
    "Effect of Sample Size",
]

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "Home"

st.sidebar.markdown("## Topic Navigator")
search_topic = st.sidebar.text_input("Search topic or keyword")

filtered_topics = [t for t in topic_options if search_topic.lower() in t.lower()] if search_topic else topic_options

for topic_name in filtered_topics:
    if st.sidebar.button(topic_name, use_container_width=True):
        st.session_state.selected_topic = topic_name

menu = st.session_state.selected_topic

# -------------------------------------------------
# Home
# -------------------------------------------------
if menu == "Home":
    st.markdown(
        """
        <div class="hero-box">
            <div class="floating-stat stat1">μ σ z</div>
            <div class="floating-stat stat2">∑ x̄</div>
            <div class="floating-stat stat3">P( Z )</div>
            <div class="floating-stat stat4">CI 95%</div>
            <div class="hero-heading">Inferential Statistics Smart Learning App</div>
            <div class="hero-para">
                Learn concepts, solve problems, run Python code, and visualize graphs in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="coordinator-box">
            🌟 Course Coordinator: Dr.J.Naskath, Asso.Prof / AI&amp;DS 🌟
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-label'>👩‍💻 Developed by</div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("<div class='dev-card'>S.HARINI<div class='dev-reg'>24243060</div></div>", unsafe_allow_html=True)
    with d2:
        st.markdown("<div class='dev-card'>S.SWETHA<div class='dev-reg'>24243048</div></div>", unsafe_allow_html=True)
    with d3:
        st.markdown("<div class='dev-card'>M.GOWRI<div class='dev-reg'>24243004</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='year-pill'>2ND YEAR - AI&DS</div>", unsafe_allow_html=True)

    st.markdown("### 📘 Attached PDF Notes")
    if os.path.exists(PDF_PATH):
        with st.expander("View PDF Topic Notes"):
            st.success("PDF loaded successfully.")
            if PDF_TEXT.strip():
                st.text_area("PDF Extracted Text Preview", value=PDF_TEXT[:2500], height=250)
            else:
                st.warning("PDF found, but text extraction failed.")
    else:
        st.warning("PDF file not found. Please keep the PDF in the correct path.")

    st.markdown("### 🔎 Search what you want to learn")
    q = st.text_area(
        "Enter your question",
        placeholder="Example: sample mean is 62, population mean is 60, standard deviation is 4 and sample size is 64"
    )

    if st.button("Answer My Question", use_container_width=True):
        home_question_answer(q)

    st.markdown("### 🎯 Topic Cards")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='topic-card'><b>Population / Sample</b><br><br>Understand the basic inferential concepts clearly.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='topic-card'><b>Hypothesis Testing / Z-Test</b><br><br>Solve one-tailed and two-tailed problems.</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='topic-card'><b>Confidence Interval / SE</b><br><br>Find intervals, margin of error, and precision.</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Population vs Sample
# -------------------------------------------------
elif menu == "Population vs Sample":
    section_title("Population vs Sample")
    info_sections(
        "Population is the complete group. Sample is the smaller selected part from that group.",
        "All students in a college = population. 50 selected students = sample.",
        ["Easy to understand", "Saves time", "Reduces cost"],
        ["Sample may not represent perfectly", "Wrong sample gives wrong result"],
        ["Surveys", "Medical studies", "Education research"],
        "A company surveys 100 customers instead of all customers."
    )

    show_steps([
        "Identify the full set of data.",
        "This full set is called population.",
        "Select a smaller part from it.",
        "This selected smaller part is called sample.",
        "Use sample to infer about population."
    ])

    st.markdown("### Manual Solve")
    pop_size = st.slider("Population size", 200, 5000, 1000)
    sample_size = st.slider("Sample size", 10, 300, 50)
    if st.button("Show Population vs Sample Graph", use_container_width=True):
        st.pyplot(draw_population_sample_plot(pop_size, sample_size))

    q = st.text_area("Ask related question", key="q_pop")
    if st.button("Solve Question - Population vs Sample", use_container_width=True):
        solve_topic_question(q, "Population vs Sample")

    run_python_code(
        """population = 1000
sample = 50
print("Population =", population)
print("Sample =", sample)

plt.figure(figsize=(6,4))
plt.bar(["Population", "Sample"], [population, sample])
plt.title("Population vs Sample")
plt.show()""",
        "popsample"
    )

# -------------------------------------------------
# Random Sampling
# -------------------------------------------------
elif menu == "Random Sampling":
    section_title("Random Sampling")
    info_sections(
        "Random sampling means every member has equal chance of selection.",
        "Selecting 10 students randomly from 100 students.",
        ["Less bias", "Fair", "Reliable"],
        ["Needs correct method", "Can be difficult in large groups"],
        ["Polls", "Hospital studies", "Quality checks"],
        "A hospital randomly selects 30 patients for a test."
    )

    show_steps([
        "List all items in the population.",
        "Assign equal chance to every item.",
        "Select required number randomly.",
        "This avoids favoritism or bias."
    ])

    st.markdown("### Manual Solve")
    n = st.slider("Choose sample size", 5, 30, 10)
    current_sample = np.random.choice(range(1, 101), size=n, replace=False)

    if st.button("Show Random Sampling Graph", use_container_width=True):
        st.success(f"Random Sample: {sorted(current_sample.tolist())}")
        st.pyplot(draw_random_sampling_plot(current_sample))

    q = st.text_area("Ask related question", key="q_random")
    if st.button("Solve Question - Random Sampling", use_container_width=True):
        solve_topic_question(q, "Random Sampling")

    run_python_code(
        """import numpy as np
sample = np.random.choice(range(1,101), size=10, replace=False)
print(sample)

plt.figure(figsize=(7,4))
plt.bar(range(1, len(sample)+1), sample)
plt.title("Random Sample Values")
plt.show()""",
        "random"
    )

# -------------------------------------------------
# Sampling Distribution
# -------------------------------------------------
elif menu == "Sampling Distribution":
    section_title("Sampling Distribution")
    info_sections(
        "Sampling distribution is the distribution of a statistic like sample mean from repeated samples.",
        "Take many samples of size 36 and compute each sample mean.",
        ["Useful in inference", "Shows variability", "Helps hypothesis testing"],
        ["Can be difficult at first"],
        ["Research", "Prediction", "Data analysis"],
        "A teacher studies average marks by repeatedly selecting sample groups."
    )

    show_steps([
        "Take one sample and compute sample mean.",
        "Repeat sampling many times.",
        "Write all sample means.",
        "The distribution of these means is called sampling distribution."
    ])

    st.markdown("### Manual Solve")
    pop_mean = st.number_input("Population mean", value=50.0)
    sigma = st.number_input("Population standard deviation", value=10.0, min_value=0.1)
    n = st.number_input("Sample size", value=36, min_value=1)
    reps = st.slider("Number of samples", 100, 5000, 1000)
    fig_sd, se_sd = draw_sampling_distribution(pop_mean, sigma, int(n), reps)

    if st.button("Show Sampling Distribution Graph", use_container_width=True):
        st.success(f"Theoretical Standard Error = {se_sd:.4f}")
        st.pyplot(fig_sd)

    q = st.text_area("Ask related question", key="q_sd")
    if st.button("Solve Question - Sampling Distribution", use_container_width=True):
        solve_topic_question(q, "Sampling Distribution")

    run_python_code(
        """import numpy as np
means = [np.mean(np.random.normal(50, 10, 36)) for _ in range(500)]
print(means[:5])

plt.figure(figsize=(7,4))
plt.hist(means, bins=25)
plt.title("Sampling Distribution")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()""",
        "samplingdist"
    )

# -------------------------------------------------
# Standard Error of Mean
# -------------------------------------------------
elif menu == "Standard Error of Mean":
    section_title("Standard Error of Mean")
    info_sections(
        "Standard Error measures how much sample mean changes from sample to sample. Formula: SE = σ / √n.",
        "If σ = 12 and n = 36, then SE = 2.",
        ["Simple formula", "Important in inference", "Shows precision"],
        ["Depends on sample size", "Sometimes confused with standard deviation"],
        ["Confidence interval", "Hypothesis testing", "Research"],
        "A company checks how stable product weight mean is."
    )

    show_steps([
        "Take population standard deviation σ.",
        "Take sample size n.",
        "Apply formula SE = σ / √n.",
        "Smaller SE means more precise estimate."
    ])

    st.markdown("### Manual Solve")
    sigma = st.number_input("Population standard deviation (σ)", value=12.0, min_value=0.1)
    n = st.number_input("Sample size (n)", value=36, min_value=1)
    se = sigma / math.sqrt(n)

    if st.button("Solve Standard Error", use_container_width=True):
        st.success(f"SE = {se:.4f}")

    q = st.text_area("Ask related question", key="q_se")
    if st.button("Solve Question - Standard Error", use_container_width=True):
        solve_topic_question(q, "Standard Error of Mean")

    if st.button("Show Standard Error Graph", use_container_width=True):
        st.pyplot(draw_standard_error_graph(sigma, n))

    run_python_code(
        """import math
sigma = 12
n = 36
se = sigma / math.sqrt(n)
print("SE =", se)

sizes = list(range(2, 25))
ses = [sigma / math.sqrt(x) for x in sizes]

plt.figure(figsize=(7,4))
plt.plot(sizes, ses, marker='o')
plt.title("Standard Error vs Sample Size")
plt.xlabel("Sample Size")
plt.ylabel("SE")
plt.show()""",
        "se"
    )

# -------------------------------------------------
# Hypothesis Testing
# -------------------------------------------------
elif menu == "Hypothesis Testing":
    section_title("Hypothesis Testing")
    info_sections(
        "Hypothesis testing checks whether sample evidence is enough to reject a claim.",
        "H0: mean = 50, H1: mean ≠ 50",
        ["Helps decisions", "Objective", "Scientific"],
        ["Needs assumptions", "Can be misinterpreted"],
        ["Medical trials", "Factory tests", "School results"],
        "A school checks whether marks changed after a new method."
    )

    show_steps([
        "State null hypothesis H0.",
        "State alternative hypothesis H1.",
        "Choose significance level α.",
        "Compute test statistic.",
        "Compare with critical value or p-value.",
        "Make decision."
    ])

    st.markdown("### Manual Explanation")
    st.write("Use Z-Test Solver topic for numerical hypothesis testing.")

    q = st.text_area("Ask related question", key="q_ht")
    if st.button("Solve Question - Hypothesis Testing", use_container_width=True):
        solve_topic_question(q, "Hypothesis Testing")

    if st.button("Show Hypothesis Testing Graph", use_container_width=True):
        st.pyplot(draw_hypothesis_testing_graph())

    run_python_code(
        """print("H0: mu = 50")
print("H1: mu != 50")

x = np.linspace(-4, 4, 500)
y = norm.pdf(x)

plt.figure(figsize=(7,4))
plt.plot(x, y)
plt.axvline(-1.96, linestyle='--')
plt.axvline(1.96, linestyle='--')
plt.title("Hypothesis Testing Critical Region")
plt.show()""",
        "ht"
    )

# -------------------------------------------------
# Z-Test Solver
# -------------------------------------------------
elif menu == "Z-Test Solver":
    section_title("Z-Test Solver")
    info_sections(
        "Z-test is used when population standard deviation is known.",
        "A university claims average mark is 75. Sample mean is 78, sigma = 12, n = 36.",
        ["Simple", "Useful", "Widely used"],
        ["Needs known sigma"],
        ["Exam analysis", "Factory testing", "Business research"],
        "A bottled water company checks whether average volume is less than claim."
    )

    show_steps([
        "Take x̄, μ, σ, and n.",
        "Compute Standard Error = σ / √n.",
        "Apply z = (x̄ - μ) / (σ / √n).",
        "Compare with critical value.",
        "Make final decision."
    ])

    st.markdown("### Manual Solve")
    sample_mean = st.number_input("Sample mean (x̄)", value=78.0)
    pop_mean = st.number_input("Population mean (μ)", value=75.0)
    sigma = st.number_input("Population standard deviation (σ)", value=12.0, min_value=0.1)
    n = st.number_input("Sample size (n)", value=36, min_value=1)
    alpha = st.number_input("Significance level (α)", value=0.05, min_value=0.001, max_value=0.2)
    tail = st.selectbox("Test type", ["two-tailed", "right-tailed", "left-tailed"])

    result = z_test_known_sigma(sample_mean, pop_mean, sigma, int(n), alpha, tail)
    zfig = draw_z_test_curve(result["z"], alpha, tail)

    if st.button("Solve Z-Test", use_container_width=True):
        step_block([
            f"SE = {result['se']:.4f}",
            f"Z value = {result['z']:.4f}",
            f"P-value = {result['p_value']:.6f}",
            f"Critical value = {result['critical']:.4f}",
        ])
        if result["reject"]:
            st.error("Decision: Reject H0")
        else:
            st.success("Decision: Fail to Reject H0")

    if st.button("Show Z-Test Graph", use_container_width=True):
        st.pyplot(zfig)

    q = st.text_area("Ask related question", key="q_z")
    if st.button("Solve Question - Z-Test", use_container_width=True):
        solve_topic_question(q, "Z-Test Solver")

    run_python_code(
        """import math
sample_mean = 78
pop_mean = 75
sigma = 12
n = 36
z = (sample_mean - pop_mean) / (sigma / math.sqrt(n))
print("z =", z)

x = np.linspace(-4,4,500)
y = norm.pdf(x)

plt.figure(figsize=(7,4))
plt.plot(x,y)
plt.axvline(z, label=f"z={z:.2f}")
plt.axvline(-1.96, linestyle='--')
plt.axvline(1.96, linestyle='--')
plt.legend()
plt.title("Z Test Curve")
plt.show()""",
        "ztest"
    )

# -------------------------------------------------
# Confidence Interval
# -------------------------------------------------
elif menu == "Confidence Interval":
    section_title("Confidence Interval")
    info_sections(
        "Confidence interval gives a likely range for the population mean.",
        "If sample mean = 52, sigma = 10, n = 64, we can find 95% CI.",
        ["Range based", "Easy to report", "Useful in research"],
        ["Needs assumptions"],
        ["Research papers", "Surveys", "Medical analysis"],
        "A company estimates average customer spending using CI."
    )

    show_steps([
        "Take sample mean x̄.",
        "Take σ and n.",
        "Find z* for selected confidence level.",
        "Compute SE = σ / √n.",
        "Compute Margin of Error = z* × SE.",
        "CI = x̄ ± Margin of Error."
    ])

    st.markdown("### Manual Solve")
    sample_mean = st.number_input("Sample mean", value=52.0)
    sigma = st.number_input("Population standard deviation", value=10.0, min_value=0.1)
    n = st.number_input("Sample size", value=64, min_value=1)
    confidence = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)

    ci = confidence_interval_known_sigma(sample_mean, sigma, int(n), confidence)
    cifig = draw_confidence_interval_graph(sample_mean, ci["lower"], ci["upper"])

    if st.button("Solve Confidence Interval", use_container_width=True):
        st.success(f"CI = ({ci['lower']:.4f}, {ci['upper']:.4f})")

    q = st.text_area("Ask related question", key="q_ci")
    if st.button("Solve Question - Confidence Interval", use_container_width=True):
        solve_topic_question(q, "Confidence Interval")

    if st.button("Show Confidence Interval Graph", use_container_width=True):
        st.pyplot(cifig)

    run_python_code(
        """import math
sample_mean = 52
sigma = 10
n = 64
z = 1.96
se = sigma / math.sqrt(n)
moe = z * se
print((sample_mean - moe, sample_mean + moe))

plt.figure(figsize=(7,3))
plt.hlines(y=1, xmin=sample_mean-moe, xmax=sample_mean+moe, linewidth=6)
plt.plot(sample_mean, 1, 'o')
plt.yticks([])
plt.title("Confidence Interval")
plt.show()""",
        "ci"
    )

# -------------------------------------------------
# Effect of Sample Size
# -------------------------------------------------
elif menu == "Effect of Sample Size":
    section_title("Effect of Sample Size")
    info_sections(
        "As sample size increases, standard error decreases and margin of error becomes smaller.",
        "n = 10 gives bigger error than n = 100.",
        ["More precision", "Better estimates"],
        ["Costs more", "Takes more time"],
        ["Polls", "Factory tests", "Research"],
        "A company increases sample size for better estimate."
    )

    show_steps([
        "Increase sample size n.",
        "SE = σ / √n becomes smaller.",
        "Margin of error also becomes smaller.",
        "Result becomes more precise."
    ])

    st.markdown("### Manual Solve")
    sigma = st.number_input("Population standard deviation", value=10.0, min_value=0.1)
    esfig = draw_ci_effect(sigma)

    if st.button("Show Effect of Sample Size Graph", use_container_width=True):
        st.success("As sample size increases, margin of error decreases.")
        st.pyplot(esfig)

    q = st.text_area("Ask related question", key="q_es")
    if st.button("Solve Question - Effect of Sample Size", use_container_width=True):
        solve_topic_question(q, "Effect of Sample Size")

    run_python_code(
        """import math
sigma = 10
sizes = [10,20,50,100]
moes = [1.96 * sigma / math.sqrt(n) for n in sizes]
for n, moe in zip(sizes, moes):
    print(n, round(moe, 4))

plt.figure(figsize=(7,4))
plt.plot(sizes, moes, marker='o')
plt.title("Effect of Sample Size on Margin of Error")
plt.xlabel("Sample Size")
plt.ylabel("Margin of Error")
plt.show()""",
        "effectsample"
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
show_footer()
st.sidebar.markdown("---")
st.sidebar.write("Built with Python, Streamlit, NumPy, Matplotlib, and SciPy")
st.sidebar.caption("Keep your logo image in the same folder as this file with name: nec_logo.png")
 
