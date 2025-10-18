# Derivative Securities (MSc course)

**Materials to share with MSc in Quantitative Finance and MSc in Financial Mathematics students.**

This repository contains supplementary materials and example codes related to derivative pricing and stochastic volatility models.  
It’s meant purely for *extra learning* and exploration — not examinable content — but it may also be helpful for developing intuition useful in research or quantitative finance job interviews.

---

## Contents

### `Theory/`
This folder includes a few concise write-ups (PDFs) with step-by-step derivations of key option pricing models:

- **Black and Scholes (1973)** – The classical constant-volatility model.  
- **Heston (1993)** – Stochastic volatility model with mean reversion and correlation between the asset and its variance.  
- **Bates-style model (1996)** – A heuristic extension of Heston that introduces jumps in asset returns.  
- **Dupire (1994)** – Local volatility framework linking implied volatilities to option prices.  
- **Derman, Ergener, and Kani (1995)** – Static replication and interpretation of the volatility surface.

These notes are intended to give students an intuitive understanding of how these models work in practice, how they connect to one another, and how volatility modeling evolved historically.  
They are not meant to be exhaustive or formally precise. All errors and typos are mine alone.

---

### `Codes/`
This folder includes working C++ implementations for:

- **European and American option pricing under Heston (1993)** using characteristic functions and numerical integration.  
- **Monte Carlo simulations** of the Heston model using the Andersen's (2008) *Quadratic Exponential (QE)* scheme.

Only Heston is implemented directly.  
However, with simple extensions you can:
- Obtain the **Bates (1996)** model by adding jumps (simply adjust the characteristic functions, and add relevant parameters), or  
- Recover **Black–Scholes (1973)** by setting stochastic variance terms to zero (there are simpler ways to obtain BS prices though).

---

## A note on parameters
Originally, these functions were written for research purposes, so you’ll see **two parameters** (`xi` and `sigma`) contributing to the variance process — representing **aggregate** and **idiosyncratic** components (what I called "productivity shocks").  
For your purposes, you can safely **set the idiosyncratic parts to zero** — it will behave like a standard Heston implementation.  
> (Yes, I know... I was lazy to rename everything)

In this setup, the “shock” drives both variance and, effectively, the underlying price dynamics.

---

## Purpose and disclaimer
These materials are shared purely for educational purposes, ie to help MSc students:
- Build intuition on stochastic volatility and practical option pricing,
- See how theory connects to implementation,
- And hopefully find these insights useful in the job market.

They are *not* examinable material, nor do they replace formal lectures.  
Everything here was written and coded by me; there may be mistakes, typos, or unpolished bits — please use it with understanding.

If you spot any errors or have suggestions, feel free to **reach out or open an issue**.  
All errors are my own — and all curiosity is welcome.

---

*Prepared and maintained by*  
**Sergey Mazyavkin, Ph.D. in Finance Candidate at the Alliance Manchester Business School**
