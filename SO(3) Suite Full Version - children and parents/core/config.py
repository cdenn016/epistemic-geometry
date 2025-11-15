import numpy as np
import sys

# ===========================
# DEBUGGING AND UTILITIES
# ===========================


def set_config_from_params(params):
    """Dynamically override config attributes from a dictionary."""
    this_module = sys.modules[__name__]
    for k, v in params.items():
        setattr(this_module, k, v)


enable_xscale = False
use_parent_cg = False
cg_backend = "loky"
parent_update_mode = "off"


phi_morphism_type       = "gauge_covariant"         # identity  gauge_covariant 
phi_model_morphism_type = "gauge_covariant"

morphism_normalize    = True



cg_phi_mode        = "auto"     # "auto" | "exact" | "fast"
cg_phi_model_mode  = "auto"

agent_seed    = 2435 

# Toggle between diagonal and full Σ initialization
diagonal_sigma = False

parent_cg_period = 250


# ===========================
# DOMAIN / SPATIAL SETTINGS
# ===========================
K_q           = 3
K_p           = 5      # Fiber dimension of belief/model space (depends on representation)
                 
D             = 2           #dimension of base manifold - d=2 validated

L             = 25         #length of hypercubic base-manifold - PBCs

steps         = 250

N             = 4                      # Number of agents
max_neighbors = 4        # Max number of agent neighbors

n_jobs = N

agent_radius_range = (4,4)         #agent radii
fixed_location = True 


 
# ===========================
# COUPLINGS & PENALTIES
# ===========================

alpha                  = 1
beta                   = 1
belief_mass            = 1e-5
curvature_weight       = 1e-4

feedback_weight         = 0
beta_model              = 1
model_mass              = 1e-5
model_curvature_weight  = 1e-4

entropy_weight          = 0


#==========================
#    IF NEEDED/CURIOUS
#==========================


phi_coupling_weight           = 0             # set > 0 to enable
phi_coupling_mode             = "projection"  # or "conjugation"


fisher_geometry_weight        = 0      
fisher_model_geometry_weight  = 0

# ===========================
# INITIALIZATION
# ===========================

# ─── Belief Bundle (q) Parameters ───
q_mu_range                     = (-1.0, 1.0)              # Range for μ_q
q_sigma_range                  = (0.2, 1.0)           # Diagonal Σ_q entries

belief_noise_scale_mu          = 0.01         # Additive noise for μ_q
belief_noise_scale_sigma       = 0.01      # Additive noise for Σ_q


# ─── Model Bundle (p) Parameters ───
p_mu_range                     = (-1.0, 1.0)             # Range for μ_p
p_sigma_range                  = (0.2, 1.0)           # Diagonal Σ_p entries

model_noise_scale_mu           = 0.01          # Additive noise for μ_p
model_noise_scale_sigma        = 0.01       # Additive noise for Σ_p

sigma_outside_val              = 1e3  # large uncertainty outside support




#===========================================================================
#
#                            EMERGENCE
#
#============================================================================
# ===== CORE =====
support_tau = 1e-3            # universal floor for mask->bool

# Parent “cores” (used only where core logic is required)
core_abs_tau = 0.10
core_rel_tau = 0.50

# ===== DETECTOR / SPAWN =====
detector_period   = 1
seed_min_kids     = 2
emerge_min_area   = 3          # was 2; 8 is stabler
# Auto gate: aim for ~N new parents each step (replaces percentile knobs)
target_new_parents_per_step = 2

# spawn essentials
spawn_min_cover2_px = 2
spawn_nms_iou       = 0.25
spawn_min_dist_px   = 2.0
parent_max_new_per_step = 4

# ===== AGREEMENT MAPS (for coalition & detector) =====
blend_qp_alpha = 0.5
tau_align_q    = 0.30
tau_align_p    = 0.30
build_model_agreement_maps = True   # set True if you want Ap_agg too

# ===== ASSIGNMENT (light; reconcile owns hysteresis) =====
parent_assign_min_overlap_px = 8
parent_assign_iou_add_thr    = 0.15  # KEEP lower add than keep (real hysteresis)
parent_assign_iou_keep_thr   = 0.05
assign_child_erode_iters     = 1     # erode child masks before IoU

# ===== RECONCILE / PARENT DYNAMICS =====
parent_freeze_steps          = 3000000
parent_orphan_grace_steps    = 10


parent_active_level          = None  # None => auto (percentile-based)
parent_active_level_q        = 60.0  # percentile for auto-threshold
parent_active_level_rel      = 0.60
parent_active_level_min      = 0.06
parent_active_thr_beta       = 0.30  # EMA smoothing of active threshold
parent_eta_up                = 0.20
parent_eta_down              = 0.15
parent_delta_cap             = 0.12
parent_feather_sigma         = 0.8
parent_commit_eps            = 1e-4
parent_active_min_cc_px      = 8
parent_mask_change_tol       = 5e-3  # drives CG “changed_masks”
# Coalition growth (still inside strict overlap gates)
coalition_cover_n            = 2
coalition_high_tau           = 0.18
coalition_low_tau            = 0.06
# CC stability
cc_switch_margin_u           = 3     # +px overlap w/ union to switch CC
cc_switch_persist            = 2     # require persistent win





parent_ramp_steps            = 0




# ===== COARSE-GRAINING (φ / Σ) =====
cg_eps            = 1e-6
cg_min_weight     = 8.0       # skip CG if total weight below this
cg_phi_iters      = 6
cg_phi_tol        = 1e-7
cg_sigma_eig_clip = True
cg_sigma_eig_lo   = 1e-6
cg_sigma_eig_hi   = 1e+2

# execution (leave defaults unless batching)

cg_jobs    = 1
cg_batch   = 1
cg_prefer  = None
cg_omp_thr = 1

# ===== DEBUG / DIAGNOSTICS =====




debug_invariants     = True   # assert masks ⊆ (≥2-cover ∩ assigned-union)
debug_health         = False
debug_assign_summary = False
debug_strict         = False


debug_phase_timing = False
debug_grad_timing = False





# 9) Debug
debug_detector_log       = True
debug_spawn_log          = True
debug_spawn_log_details  = True
debug_strict             = True

debug_parent_morphism_init = True
#======================
#
#   LEARNING RATES
#
#======================
tau_phi                 = 1e-4
tau_phi_model           = 1e-4    
tau_mu_belief           = 1e-5       # update rate for η₁_q
tau_sigma_belief        = 1e-6       # update rate for η₂_q
tau_mu_model            = 1e-5      # update rate for η₁_p
tau_sigma_model         = 1e-6       # update rate for η₂_p

phi_init_smooth_sigma   = 1.5
phi_init                = 0.1
phi_model_offset        = 0.05

eta_ema_alpha                   = 0.6          # try 0.4–0.8
eta_phi_adaptive_scaling        = 1.0
eta_phi_model_adaptive_scaling  = 1.0
eta_sigma_condition_scaling     = 0.1   # start small
sigma_eig_floor                 = 1e-6
sigma_cond_cap                  = 1e4

sigma_eig_cap                   = None          # usually None
sigma_trace_target              = None     # or K if you want trace control

eta_phi_min           = 1e-6
eta_phi_model_min     = 1e-6





#==========================================
#
#       META-AGENTS
#
#==========================================
# Cross-scale toggles


# Λ terms
lambda_env_down_q = 0
lambda_env_down_p = 0
lambda_env_up_q   = 0
lambda_env_up_p   = 0


# Λ on q/p
lambda_phi_env_down_q       = 0
lambda_phi_env_up_q         = 0
lambda_phi_env_down_p       = 0
lambda_phi_env_up_p         = 0



# Θ terms
lambda_env_down_theta_q = 0
lambda_env_down_theta_p = 0
lambda_env_up_theta_q   = 0
lambda_env_up_theta_p   = 0

# Θ cross-fiber
lambda_phi_env_down_theta_q = 0
lambda_phi_env_up_theta_q   = 0
lambda_phi_env_down_theta_p = 0
lambda_phi_env_up_theta_p   = 0





# ===========================
# AGENT GEOMETRY
# ===========================
gaussian_blur_range_morphism = (1,2)
mu_clip = 3.0

#================================
#
#  INITIALIZE GLOBAL FIELDS 
#      WITH GENTLE NOISE
#================================
use_global_belief_template = False
use_global_model_template  = False

identical_models = False         # If True, all agents share the same p, mu_p_field, sigma_p_field


# ===========================
# STABILITY AND CLIPPING
# ===========================
   
gradient_clip_phi         = -1     # clip ‖∇φ‖ per pixel to ≤ 1.0
gradient_clip_phi_model   = -1     # same for φ̃

energy_eps = 1e-6
grid_dx = 1
# ===========================
# EPSILONS & NUMERICAL STABILITY
# ===========================

log_eps            = 1e-6     # Stabilize log(x)
eps                = 1e-5
# ===========================
# MASK & INTERACTION THRESHOLDS
# ===========================
support_tau        = 1e-3
support_cutoff_eps = 1e-3    # Mask threshold to include point in agent support################################
overlap_eps        = 1e-3     # Threshold for computing inter-agent overlap


# ===========================
# CHECKPOINTING
# ===========================

checkpoint_dir      = "checkpoints"
precision = np.float32

checkpoint_interval = 1

domain_size = (L,) * D
num_cores = 1  # or however many threads you want to use
group_name = 'so3'               # Options: "u1", "so3", "so3", 'su2'.

phi_bch_max_norm = 25.0
# Enable numerical safety logging
debug = True  # Global debug flag for extra logging inside φ updates                       
fail_on_fallback = False                # Allow recovery rather than crashing


#=================
#    LOGDET
#=================

# ── Numerical Safeguards ─────────────────────────────
max_fisher_condition = 1e5   # Maximum allowed condition number before fallback


# Control fallback thresholds (optional)
max_safe_inv_fallbacks = 1000
max_safe_logdet_fallbacks = 1000

# ─────────────────────────────────────────────────────────
# Logging / dump toggles (matches new logger + field dumper)
# ─────────────────────────────────────────────────────────

periodic_x = True          # set per your sim
periodic_y = True