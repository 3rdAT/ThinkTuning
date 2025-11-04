# from math import cos, pi

# def build_scheduler(cfg):
#     t = 0
#     def constant():                return cfg.init_pct
#     def exp():       nonlocal t; v = max(cfg.min_pct, cfg.init_pct * cfg.decay_rate ** t); t += 1; return v
#     def linear():    nonlocal t; v = max(cfg.min_pct, cfg.init_pct - (cfg.init_pct-cfg.min_pct) * t/cfg.decay_steps); t += 1; return v
#     def cosine():    nonlocal t; v = cfg.min_pct + 0.5*(cfg.init_pct-cfg.min_pct)*(1+cos(pi*min(t,cfg.decay_steps)/cfg.decay_steps)); t += 1; return v
#     table = {"constant": constant, "exp": exp, "linear": linear, "cosine": cosine}
#     return table[cfg.type.lower()]

from math import cos, pi, pow

def build_scheduler(cfg):
    """
    Returns a callable f() → pct.
    Required cfg fields
        type          : "constant" | "exp" | "linear" | "cosine"
        init_pct      : float, starting value
        min_pct       : float, value after the decay period
        num_train_steps : int, total optimisation steps
    Optional
        eps           : float, treat as “effectively zero” for exp (default 1e-8)

    All schedules decay until step ⌊num_train_steps / 4⌋ and then remain at min_pct.
    """
    quarter_steps = max(1, cfg.num_train_steps // 3)      # T/N
    eps = getattr(cfg, "eps", 1e-8)                       # ≈0 for the exp curve

    # ---- derive γ so that init_pct * γ^{quarter_steps} = eps  ----------------
    if cfg.type.lower() == "exp":
        gamma = pow(max(eps, cfg.min_pct) / cfg.init_pct, 1.0 / quarter_steps)

    t = 0  # internal step counter ------------------------------------------------

    print("The scheduler has been built!")

    def constant():
        print("The constant scheduler has been called!")
        return cfg.init_pct

    def exp():
        print("The exp scheduler has been called!")
        nonlocal t
        v = cfg.init_pct * (gamma ** min(t, quarter_steps))
        t += 1
        return max(cfg.min_pct, v) if t <= quarter_steps else cfg.min_pct

    def linear():
        print("The linear scheduler has been called!")
        nonlocal t
        v = cfg.init_pct - (cfg.init_pct - cfg.min_pct) * (min(t, quarter_steps) / quarter_steps)
        t += 1
        return v

    def cosine():
        print("The cosine scheduler has been called!")
        nonlocal t
        v = cfg.min_pct + 0.5 * (cfg.init_pct - cfg.min_pct) * (
            1 + cos(pi * min(t, quarter_steps) / quarter_steps)
        )
        t += 1
        return v

    table = {"constant": constant, "exp": exp, "linear": linear, "cosine": cosine}
    return table[cfg.type.lower()]