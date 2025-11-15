import core.config as config

def set_config_from_params(params):
    """Safely override known config attributes from a dictionary
    and then recompute all dependent quantities."""
    valid_keys = {
        k for k in dir(config)
        if not k.startswith("__") and not callable(getattr(config, k))
    }

    # 1) Override raw params into config.py
    for k, v in params.items():
        if k in valid_keys:
            setattr(config, k, v)
        else:
            print(f"[WARNING] Ignored unknown config key: '{k}'")

    
    # ───────────────────────────────────────────────────────────────────────────

    # ── Recompute spatial sizing & any K-dependent ranges ───────────────────────
    config.domain_size = (config.L,) * config.D
    config.q_mu_range = (2, config.K_q - 2)
    config.p_mu_range = (2, config.K_p - 2)
    # ───────────────────────────────────────────────────────────────────────────

    