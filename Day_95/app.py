from flask import Flask, render_template, request, send_file, redirect, url_for
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Prior (Uniform)
ALPHA0 = 1
BETA0 = 1

def parse_sequence(text):
    """
    Parse a string of 0/1 values separated by spaces/commas/newlines.
    Returns (successes, trials)
    """
    if not text:
        return None
    # allow commas or spaces
    tokens = text.replace(',', ' ').split()
    vals = []
    for t in tokens:
        if t.strip() == '':
            continue
        try:
            v = int(t)
            if v not in (0, 1):
                return None
            vals.append(v)
        except:
            return None
    if len(vals) == 0:
        return None
    return sum(vals), len(vals)

def compute_stats(S, N, alpha0=ALPHA0, beta0=BETA0):
    """Returns dict of MLE and Bayesian posterior stats."""
    S = int(S)
    N = int(N)
    mle = S / N if N > 0 else None

    alpha = alpha0 + S
    beta_ = beta0 + (N - S)

    # posterior mean
    post_mean = alpha / (alpha + beta_)
    # posterior MAP: (alpha-1)/(alpha+beta-2) if denom>0; else fall back to mean
    denom = (alpha + beta_ - 2)
    if denom > 0:
        post_map = (alpha - 1) / denom
    else:
        post_map = post_mean

    # 95% credible interval
    lower = beta.ppf(0.025, alpha, beta_)
    upper = beta.ppf(0.975, alpha, beta_)

    return {
        "S": S,
        "N": N,
        "MLE": mle,
        "alpha": alpha,
        "beta": beta_,
        "posterior_mean": post_mean,
        "posterior_map": post_map,
        "ci_lower": lower,
        "ci_upper": upper
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_seq = ""
    successes = ""
    trials = ""
    if request.method == 'POST':
        # two input modes: sequence OR numeric successes & trials
        input_seq = request.form.get('sequence', '').strip()
        successes = request.form.get('successes', '').strip()
        trials = request.form.get('trials', '').strip()

        if input_seq:
            parsed = parse_sequence(input_seq)
            if parsed is None:
                result = {"error": "Invalid sequence. Use only 0 and 1, separated by spaces or commas."}
            else:
                S, N = parsed
                result = compute_stats(S, N)
        elif successes != "" and trials != "":
            try:
                S = int(successes)
                N = int(trials)
                if S < 0 or N <= 0 or S > N:
                    raise ValueError("Invalid numeric values.")
                result = compute_stats(S, N)
            except Exception as e:
                result = {"error": f"Invalid input: {e}"}
        else:
            result = {"error": "Provide either a sequence of 0/1 or both successes and trials."}
    return render_template('index.html', result=result, input_seq=input_seq, successes=successes, trials=trials)

@app.route('/posterior_plot')
def posterior_plot():
    """
    Expects query args: alpha, beta
    e.g. /posterior_plot?alpha=3&beta=5
    Returns a PNG plot of the Beta distribution
    """
    try:
        alpha = float(request.args.get('alpha', ALPHA0))
        beta_ = float(request.args.get('beta', BETA0))
    except:
        alpha = ALPHA0
        beta_ = BETA0

    x = np.linspace(0, 1, 500)
    y = beta.pdf(x, alpha, beta_)

    plt.figure(figsize=(6,3.5))
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.2)
    plt.title(f'Beta Posterior (alpha={int(alpha)}, beta={int(beta_)})')
    plt.xlabel('p (conversion rate)')
    plt.ylabel('Density')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
