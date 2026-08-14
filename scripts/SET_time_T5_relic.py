"""
SET time campaign, T5: relic computation at the 469 MeV window.
Priority 1 of the 2026-08-11 addendum. Question posed there: computable
abundance vs the derived dark matter density; candidate new registered
prediction.

Dependencies: SET_time_T4_wilson.py (recovered verbatim from the prior
chat, rerun green this session). Conditional on everything T4 is
conditional on: A1+A2, the phased-sum emission model, candidate A's
identifications (spectral unit = m_p, tick = proton Compton time).

Standard-physics inputs, flagged: eta_B = 6.1e-10; Omega_DM h^2 = 0.120;
Omega_b h^2 = 0.0224 (framework derives the RATIO to 1.4%, and the ratio
is all that enters); hot-relic formula Omega h^2 = m/(93.14 eV); entropy
degrees of freedom g*s(T0) = 3.91, g*s(170 MeV) = 17.25 (T4's value;
sensitivity band to 61.75 shown); Lyman-alpha warm-DM floor ~ few keV
(external observational input, order-of-magnitude use only).

Inference, flagged: f_kin, the fraction of QCD-epoch decoherence events
kinematically able to source a 469 MeV emission, is modeled as a
Boltzmann-weighted band 0.03 to 0.5. The verdict is insensitive across
the band and across three orders around it, shown below.

Structure: R1 target budget; R2 production at the window, two routes;
R3 the relic today; R4 the required-coupling map; R5 the thermal
ceiling; verdict.
"""
import numpy as np

ok = True
def check(name, cond):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name)
    ok = ok and cond
def obs(s): print("[OBS ] " + s)

# constants and framework numbers
hbar   = 6.582119569e-16    # eV s
m_p    = 9.3827208816e8     # eV
d      = 14.0100
T_pass = 2*np.pi*hbar/m_p   # s
g2_G   = np.exp(-2*np.pi*d) # registry benchmark coupling
m_gate0 = 0.0783            # eV, gate ground state (imported-exponent value)
m_gate1 = 57.0              # eV, gate first excited
E_res  = 0.5*m_p            # 469 MeV, lowest surviving resonance

# standard inputs
eta_B   = 6.1e-10
Odm_h2  = 0.120
Ob_h2   = 0.0224
T0      = 2.348e-4          # eV, CMB today
T_QCD   = 170e6             # eV
gs0, gsQ = 3.91, 17.25
gsQ_hi   = 61.75            # pre-transition sensitivity value

# ---------------- R1: the target budget ----------------
ratio_DM = Odm_h2/Ob_h2                      # framework derives this to 1.4%
E_budget = ratio_DM*m_p                      # eV of DM mass per baryon
print(f"      DM-to-baryon mass ratio {ratio_DM:.2f}; budget "
      f"{E_budget:.2e} eV of dark matter mass riding with every baryon")
N_need_res   = E_budget/E_res                # quanta/baryon if each is 469 MeV of MASS
N_need_gate0 = E_budget/m_gate0              # if the surviving mass is the gate ground state
print(f"      quanta needed per baryon: {N_need_res:.1f} at 469 MeV rest mass, "
      f"{N_need_gate0:.1e} at {m_gate0} eV rest mass")

# ---------------- R2: production at the window, two routes ----------------
# Route 1 (ceiling, f_kin-free): every decoherence event across the QCD
# epoch emits with probability g^2, resonant or not. This upper-bounds any
# resonant subset because coherence is dead there (T4 W3: protection O(1)),
# so resonance confers no multi-pass enhancement at the only epoch where
# the channel is kinematically open.
def n_gamma(T_eV):
    hbarc = 1.973269804e-5
    return 0.24*(T_eV/hbarc)**3
c_cm  = 2.99792458e10
t_QCD = 2e-5
Gam_Q = n_gamma(T_QCD)*4e-26*c_cm
N_dec = min(1/T_pass, Gam_Q)*t_QCD
N_ceiling = g2_G*N_dec
# Route 2 (resonant subset with the kinematic weight)
f_kin_band = (0.03, 0.5)
N_res_band = tuple(N_ceiling*f for f in f_kin_band)
print(f"      route 1 ceiling: {N_ceiling:.1e} quanta/baryon "
      f"(matches T4's 2.2e-20)")
print(f"      route 2 resonant subset: {N_res_band[0]:.1e} to "
      f"{N_res_band[1]:.1e} quanta/baryon across the f_kin band")
check("R2: the two routes agree within the f_kin band and both sit "
      ">19 orders below even the 469-MeV-mass requirement of ~10.7/baryon",
      N_ceiling/N_need_res < 1e-19 and N_res_band[1] <= N_ceiling)

# ---------------- R3: the relic today ----------------
# A quantum emitted at 469 MeV total energy with gate-scale rest mass is
# ultra-relativistic; its momentum redshifts with entropy dilution.
def p_today(gsQ_use):
    return E_res*(gs0/gsQ_use)**(1/3)*(T0/T_QCD)
p0_lo, p0_hi = p_today(gsQ_hi), p_today(gsQ)
print(f"      momentum today: {p0_lo:.1e} to {p0_hi:.1e} eV "
      f"(g*s sensitivity band); vs rest mass {m_gate0} eV")
check("R3: the relic went nonrelativistic long ago (p_today << m_gate0), "
      "so it counts as mass today, at the ground-state value",
      p0_hi < 0.1*m_gate0)
rho_relic = N_ceiling*m_gate0                # eV per baryon, ceiling case
frac = rho_relic/E_budget
print(f"      relic mass density: {rho_relic:.1e} eV/baryon = "
      f"{frac:.1e} of the DM budget")
check("R3: KILL 1 (production): at the benchmark coupling the window "
      "supplies < 1e-29 of the dark matter", frac < 1e-29)
# sanity vernacular: number density today
n_b0 = 2.5e-7                                 # baryons per cm^3, standard
n_relic = N_ceiling*n_b0                      # per cm^3
km3 = 1e15                                    # cm^3
print(f"      that is ~{n_relic*km3:.0e} quanta per cubic kilometer, "
      f"about one quantum per {1/(n_relic*km3):.0e} km^3 of space")

# ---------------- R4: required-coupling map ----------------
# What g^2 would make the gate sector the dark matter, per carrier mass?
g2_bar = 1.3e-19                              # T4 thermalization bar
for label, m_c, N_need in (
    ("469 MeV rest mass (hypothetical resonant-mass state)", E_res, N_need_res),
    (f"{m_gate1} eV first excited state", m_gate1, E_budget/m_gate1),
    (f"{m_gate0} eV ground state", m_gate0, N_need_gate0)):
    g2_req = N_need/N_dec
    zone = "ABOVE the thermalization bar (excluded thermal zone)" \
           if g2_req > g2_bar else "below the bar"
    print(f"      {label}: needs g^2 ~ {g2_req:.1e}, "
          f"{g2_req/g2_bar:.0e} x the bar, {zone}")
check("R4: KILL 2 (coupling map): every carrier mass from 0.0783 eV to "
      "469 MeV requires g^2 at least ~10 x the thermalization bar, i.e. "
      "inside the thermal zone T4 already excluded via Delta N_eff; no "
      "allowed coupling reaches the DM budget",
      (N_need_res/N_dec) > 10*g2_bar)

# ---------------- R5: the thermal ceiling ----------------
# Even granting full thermalization (which Fork 1's exclusion forbids):
# a light thermal relic obeys Omega h^2 = m / 93.14 eV per fermionic dof
# (standard hot-relic result, one dof pair; tower multiplicity would not
# change the orders).
Oh2_gate0 = m_gate0/93.14
Oh2_gate1 = m_gate1/93.14
print(f"      full-thermalization ceiling: ground state Omega h^2 = "
      f"{Oh2_gate0:.1e} = {Oh2_gate0/Odm_h2:.1e} of the DM density; "
      f"57 eV state Omega h^2 = {Oh2_gate1:.2f} = "
      f"{Oh2_gate1/Odm_h2:.1f} x the DM density")
check("R5: KILL 3a (mass): the ground state, even fully thermalized, "
      "supplies < 1% of the dark matter", Oh2_gate0/Odm_h2 < 0.01)
obs("KILL 3b (structure): the 57 eV state at thermal abundance would "
    "overshoot the DM density 5x AND sits ~2 orders below the Lyman-alpha "
    "warm-DM floor (~ few keV, external input): excluded as the dominant "
    "DM by free-streaming regardless of abundance, and reaching thermal "
    "abundance requires the Fork-1 coupling zone already excluded")

# ---------------- verdict ----------------
print("""
VERDICT (T5): the 469 MeV window does not mint a dark matter prediction.
The kill is overdetermined, three independent ways:
  1. Production: at the benchmark coupling the window fills ~3e-31 of the
     budget; coherence is dead at the only epoch the channel is open, so
     resonance confers no enhancement (T4 W3).
  2. Coupling map: any coupling large enough to reach the budget sits
     inside the thermal zone excluded by Delta N_eff (T4 W4), for every
     candidate carrier mass across 10 orders of magnitude.
  3. Carrier mass: the known gate levels are hot-relic light; the ground
     state maxes out below 1% of the DM density even at full thermal
     abundance, and the 57 eV state is free-streaming excluded.
Consequence: the gate sector is cosmologically empty at the benchmark and
cannot be the dark matter at any allowed coupling. The framework's derived
DM-to-baryon ratio (1.4% postdiction) now has NO carrier inside the
framework: NEW OPEN PROBLEM, replaces priority 1. The 'candidate
primordial production window' observation resolves: window open, pump off.
Positive residue: the framework PREDICTS a null, ~1e-20 quanta/baryon at
~1e-4 eV momenta today, invisible by construction; and the E80 unit audit
(priority 2) is now the sole load-bearing support under candidate A.
""")
print("All committed checks resolved." if ok else "FAILURES PRESENT.")
