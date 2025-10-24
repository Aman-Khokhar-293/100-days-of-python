## How it works (short)
- Input data → compute S = #successes, N = #trials
- MLE: p_hat = S / N
- Prior: Beta(alpha0=1, beta0=1)
- Posterior: Beta(alpha = alpha0 + S, beta = beta0 + N - S)
- Posterior mean = alpha / (alpha + beta)
- Posterior MAP = (alpha - 1) / (alpha + beta - 2) (if alpha, beta > 1; otherwise handle edge cases)
- 95% credible interval from posterior quantiles

---

## Run locally

1. Create virtualenv & install:
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run app:
```bash
python app.py
```

3. Open `http://127.0.0.1:5000/` in your browser.

---

## Notes
- Uses a **Uniform prior Beta(1,1)** (no prior assumptions).
- Posterior plot generated on the fly and returned as an image.