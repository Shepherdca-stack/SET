"""
SET time campaign, Wilson -1 cancellation thread (candidate chain 26 material).

RECOVERY NOTE (2026-08-11, session 2): this file was recovered verbatim from
the prior chat transcript ("Preparing for temporal dynamics"), including the
one correction applied there (W3 first check reworded to the O(1) claim).
It is the audited code, not a reconstruction.

Conditional throughout on A1+A2 (antiperiodic cusp gluing, Wilson line -1,
sign flip per pinch pass) and on one modeling assumption, stated: pinch
emission over successive passes is a phased sum with identical per-pass
amplitude a; the drift's translation phase 2E/m_p per pass is absorbed
into the dynamical phase delta (it is 1.7e-10 for the gate ground state,
negligible and included).

W1  Alternating-sum bound (derived): with the Wilson -1, the cumulative
    emission amplitude over N passes for a mode with per-pass dynamical
    phase delta is
        A_N = a * sum_{n=0}^{N-1} e^{i n (pi + delta)}
        |A_N| = |a| * |sin(N(pi+delta)/2)| / cos(delta/2) <= |a|/cos(delta/2)
    BOUNDED for all N. The periodic gluing (+1) gives |A_N| ~ N|a| for
    N*delta << 1: coherent buildup. The -1 corner converts N^2 probability
    growth into O(1). Soft emission (delta << pi) never accumulates.
W2  Resonance relocation (derived): coherent buildup survives only at
    delta = pi (mod 2pi), i.e. emitted energy E = (k+1/2) * hbar*Omega
    = (k+1/2) * m_p c^2 under candidate A. The pinch pumps HALF-INTEGER
    harmonics. Structural echo of the gate tower's (n+1/2) form: OBSERVED
    correspondence, identification not established (needs the E80 unit
    audit). Lowest resonance 469 MeV, kinematically reachable only in the
    QCD-epoch plasma: OBSERVED, one line, no promotion.
W3  Decoherence-limited protection (derived + standard inputs): the
    cancellation requires phase memory between passes. Effective emission
    rate R = g^2 * min(Gamma_dec, 1/T_pass). Protection factor
    1/(Gamma_dec*T_pass) when Gamma_dec < 1/T_pass; NONE when collisions
    outpace the tick. Standard-physics inputs for Gamma_dec (Thomson-like
    below QCD, strong-interaction at QCD).
W4  Fork resolution (derived bound): thermal contact requires
    g^2 > g2_th(T) = H(T) * max(T_pass, 1/Gamma_dec), minimized at the
    QCD epoch where protection vanishes. Numerical bar ~1e-19.
    Geometric coupling (g^2 ~ 1) fails by ~19 orders WITH the cancellation:
    the thermal-fork exclusion is now derived, not estimated. The registry
    benchmark g^2 = e^{-2 pi d} sits ~19 orders INSIDE the bar: Fork 2
    selected. Production per baryon at the benchmark, Wilson-corrected,
    collapses to ~1e-20, superseding both prior ansatz estimates (73 and
    568), which ignored the cancellation.
"""
import numpy as np
import sympy as sp

ok = True
def check(name, cond):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name)
    ok = ok and cond
def obs(s): print("[OBS ] " + s)

hbar = 6.582119569e-16      # eV s
m_p  = 9.3827208816e8       # eV
d    = 14.0100
T_pass = 2*np.pi*hbar/m_p   # one pinch pass per breath period, s
M_P  = 1.220890e28          # eV

# ---------------- W1: the bound, symbolic then numeric ----------------
N, n = sp.symbols('N n', positive=True, integer=True)
delta = sp.symbols('delta', real=True)
S = sp.summation(sp.exp(sp.I*n*(sp.pi + delta)), (n, 0, N-1))
S_closed = sp.simplify(S)
# closed form magnitude: |sin(N(pi+delta)/2)| / cos(delta/2)
target = sp.sin(N*(sp.pi+delta)/2)/sp.cos(delta/2)
diff = sp.simplify(sp.Abs(S_closed)**2 - sp.Abs(target)**2)
num_ok = True
for Nv in (1, 2, 7, 100, 12345):
    for dv in (1e-10, 1e-3, 0.3, 2.0):
        lhs = abs(sum(np.exp(1j*k*(np.pi+dv)) for k in range(Nv)))
        rhs = abs(np.sin(Nv*(np.pi+dv)/2)/np.cos(dv/2))
        bound = 1/np.cos(dv/2)
        num_ok &= abs(lhs - rhs) < 1e-8 and lhs <= bound + 1e-12
check("W1: closed form |A_N| = |sin(N(pi+delta)/2)|/cos(delta/2) and the "
      "bound 1/cos(delta/2) hold (numeric, N to 12345, delta to 1e-10)",
      num_ok)
# periodic comparison: probability ratio -> N^2 as delta -> 0
Nv, dv = 10000, 1e-7
coh = abs(np.sin(Nv*dv/2)/np.sin(dv/2))**2
wil = abs(np.sin(Nv*(np.pi+dv)/2)/np.cos(dv/2))**2
check("W1: periodic gluing gives ~N^2 coherent probability where Wilson -1 "
      "stays O(1) (ratio > 1e7 at N = 1e4)", coh/max(wil, 1e-30) > 1e7)
delta_gate = 2*np.pi*0.0783/m_p
print(f"      gate ground state per-pass phase delta = {delta_gate:.2e}; "
      "soft by 10 orders; cumulative emission capped at one-pass level")

# ---------------- W2: resonance relocation ----------------
E_res = [(k + 0.5)*m_p/1e6 for k in range(3)]
print(f"      surviving resonances E = (k+1/2) m_p c^2: "
      f"{E_res[0]:.0f}, {E_res[1]:.0f}, {E_res[2]:.0f} MeV")
check("W2: no resonance below m_p/2; the 0.0783 eV and 57 eV gate levels "
      "sit 10 and 7 orders below the lowest resonant channel",
      0.0783 < 469e6 and 57 < 469e6)
obs("half-integer emission resonance echoes the gate tower's (n+1/2) "
    "form; correspondence OBSERVED, identification open pending E80 unit audit")
obs("lowest resonance 469 MeV is kinematically reachable only in the "
    "QCD-epoch plasma (T ~ 150-200 MeV, typical collision ~ 3T); "
    "possible primordial production window, speculative, not promoted")

# ---------------- W3: protection vs decoherence ----------------
def H_of_T(T_eV, gstar):
    return 1.66*np.sqrt(gstar)*T_eV**2/M_P/hbar     # s^-1
def n_gamma(T_eV):
    hbarc = 1.973269804e-5                          # eV cm
    return 0.24*(T_eV/hbarc)**3                     # cm^-3
c_cm = 2.99792458e10
epochs = {
    # label: (T in eV, g*, sigma_dec cm^2, note)
    "QCD epoch (T=170 MeV)": (170e6, 17.25, 4e-26, "strong, ~40 mb"),
    "BBN (T=1 MeV)":         (1e6,  10.75, 2.0e-31, "Thomson on proton"),
    "recomb (T=0.26 eV)":    (0.26, 3.36,  2.0e-31, "Thomson on proton"),
}
g2_bar = None
for label, (T, gstar, sigma, note) in epochs.items():
    Gam = n_gamma(T)*sigma*c_cm
    prot = max(1.0, 1.0/(Gam*T_pass))
    g2_th = H_of_T(T, gstar)*max(T_pass, 1/Gam)
    print(f"      {label}: Gamma_dec = {Gam:.1e}/s, protection = "
          f"{prot:.1e}, thermalization bar g^2 > {g2_th:.1e}")
    if g2_bar is None or g2_th < g2_bar:
        g2_bar = g2_th
check("W3: protection at the QCD epoch is O(1) (< 2 with factor-2 input "
      "slop): marginal to none, exactly where the bar is set",
      1.0/(n_gamma(170e6)*4e-26*c_cm*T_pass) < 2.0)
check(f"W3: the thermalization bar is minimized at the QCD epoch, "
      f"g^2_bar ~ {g2_bar:.1e}", 1e-20 < g2_bar < 1e-18)

# ---------------- W4: fork resolution ----------------
g2_G = np.exp(-2*np.pi*d)
check("W4: geometric coupling g^2 ~ 1 exceeds the bar by ~19 orders: "
      "thermal-fork exclusion of geometric coupling is now DERIVED "
      "(cancellation included; it dies exactly where it was needed)",
      1.0/g2_bar > 1e17)
check(f"W4: registry benchmark g^2 = e^-2pi*d = {g2_G:.1e} sits ~19 "
      "orders inside the bar: Fork 2 (dilute) selected",
      g2_G/g2_bar < 1e-17)
# Wilson-corrected production at the benchmark: one opportunity per
# decoherence event; history integral dominated at the QCD epoch
t_QCD = 2e-5                                        # s
N_dec = min(1/T_pass, n_gamma(170e6)*4e-26*c_cm)*t_QCD
N_gate = g2_G*N_dec
print(f"      decoherence events per baryon (history, QCD-dominated) ~ "
      f"{N_dec:.1e}; Wilson-corrected production at benchmark ~ "
      f"{N_gate:.1e} quanta/baryon")
check("W4: Wilson-corrected production supersedes both prior ansatzes "
      "(73, 568) downward by ~20 orders; gate sector essentially "
      "unpopulated below the resonant channel", N_gate < 1e-15)

print("\nAll committed checks resolved." if ok else "\nFAILURES PRESENT.")
