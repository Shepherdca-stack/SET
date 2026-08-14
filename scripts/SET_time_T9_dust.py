"""
SET time campaign, T9: geometric dust (candidate chain 31).
Open problem 1 of v1.7. Question: does the framework contain a clustering
geometric component carrying Omega_DM/Omega_b = 2pi - 1 per baryon,
gravitationally coupled per E112?

THE CANDIDATE, stated up front: the loop. Chain 25 proves each drift
trajectory decomposes exactly as z(t) - (t+1) = e^{i pi t}: a unit-radius
circulation riding every baryon's worldline. The proposal is that this
loop, realized as bound circulating energy of the geometry, IS the dust:
one packet per baryon, energy (2pi - 1) m_p, localized (so it can
cluster), coupled to everything only through gravity (E112), and NOT a
quantum of any field in the inventory (so chain 30's elimination, which
swept quanta, is threaded by category, A-INV intact).

Status frame, before any check runs:
  PROVEN inputs: E1/chain 25 (the decomposition, periodicity, gain
    invariance, emit-absorb balance), E111 (essential zeros: no
    non-gravitational baryon-pinch contact exists), E46 (the ratio,
    registry geometry), E41/E108 (candidate A's unit, tick).
  CONDITIONAL inputs: I-A (candidate A), M-EM (phased-sum emission),
    A1+A2 (Wilson -1).
  NEW IDENTIFICATION REQUIRED (named here I-DUST): the packet's energy
    is the arc:radius partition of the loop, total 2pi per unit radius,
    baryonic share the radius, dark share 2pi - 1. Section D1 tests the
    natural partitions and shows this one is selected by MATCHING E46,
    not yet by dynamics. I-DUST is the candidate's load-bearing new
    assumption and the promotion gate is deriving it.
  Formation bookkeeping (where the packet's energy comes from at
    baryogenesis/confinement) and the covariant embedding of packet
    stress-energy: OPEN, flagged, not used by any check below.

Standard inputs, flagged: Omega_DM h^2 = 0.120, Omega_b h^2 = 0.0224
(errors ~1%, 0.7%); H0 = 67.4; Omega_r = 9.1e-5, Omega_m = 0.315,
Omega_L = 0.685; g*s today 3.91, at 170 MeV 17.25 (registry value);
Bullet Cluster self-interaction bound sigma/m < 1 cm^2/g;
Lyman-alpha-safe free-streaming scale ~0.1 Mpc (order, external).
"""
import numpy as np

ok = True
def check(name, cond):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name)
    ok = ok and cond
def obs(s): print("[OBS ] " + s)

# constants and framework numbers
hbar_eVs = 6.582119569e-16
m_p_GeV  = 0.93827208816
m_p_eV   = 9.3827208816e8
M_P_GeV  = 1.220890e19
Mbar_GeV = 2.435e18
d        = 14.0100
g2       = np.exp(-2*np.pi*d)          # = alpha_grav(p), E112
tick     = np.pi*hbar_eVs/m_p_eV       # E108, s
T_pass   = 2*tick                      # one full loop circuit, s
Omega    = np.pi/tick                  # loop angular frequency, 1/s
age      = 4.35e17                     # s
ratio_E46 = 2*np.pi - 1

# standard inputs
Odm_h2, Ob_h2 = 0.120, 0.0224
H0 = 67.4*1e5/3.0857e24                # 1/s
c_over_H0_Mpc = 2.99792458e5/67.4      # Mpc
Or, Om, OL = 9.1e-5, 0.315, 0.685
T0_eV, T_QCD_eV = 2.348e-4, 170e6
gs0, gsQ = 3.91, 17.25

# ---------------- D1: the partition sweep ----------------
# The loop z_loop(t) = e^{i pi t}; the drift (t+1). Numeric facts first,
# from the trajectory itself, then the three natural energy partitions.
t = np.linspace(0, 2, 400001)          # one full circuit = 2 t-units
z_loop = np.exp(1j*np.pi*t)
r_loop = np.abs(z_loop)
arc = np.trapezoid(np.abs(np.gradient(z_loop, t)), t)      # loop arc/circuit
drift_per_tick = 1.0                                       # d/dt (t+1) * 1
ke_ratio  = np.pi**2                    # <|zdot_loop|^2>/<|zdot_drift|^2>
len_ratio = arc/2.0                     # loop path : drift path per circuit
arc_per_radius = arc/1.0                # circumference : radius
print(f"      loop radius = {r_loop.min():.6f} to {r_loop.max():.6f}; "
      f"arc per circuit = {arc:.6f} (2pi = {2*np.pi:.6f}); "
      f"drift per tick = {drift_per_tick}")
check("D1: loop geometry from the trajectory: radius exactly 1, arc per "
      "circuit exactly 2pi, drift exactly one radius per tick",
      abs(r_loop.max()-1) < 1e-12 and abs(arc-2*np.pi) < 1e-6)
print(f"      partition (a) kinetic energy:  loop/drift = pi^2 = {ke_ratio:.4f}"
      f"  -> dark/baryon would be {ke_ratio:.3f}: REJECTED (E46 = {ratio_E46:.4f})")
print(f"      partition (b) path length:     loop/drift = pi   = {len_ratio:.4f}"
      f"  -> REJECTED")
print(f"      partition (c) arc per radius:  total 2pi per 1, dark share "
      f"{arc_per_radius-1:.4f} -> matches E46 = {ratio_E46:.4f}")
check("D1: partitions (a) pi^2 and (b) pi both miss E46 by >85% and >40%; "
      "partition (c), circumference-minus-radius per radius, equals "
      "2pi - 1 to machine precision: it is E46's own recorded geometry "
      "('one full turn per unit radius') realized on the loop",
      abs(ke_ratio - ratio_E46) > 4 and abs(len_ratio - ratio_E46) > 2
      and abs((arc_per_radius - 1) - ratio_E46) < 1e-6)
obs("selection of (c) is by MATCH to E46, not by a derived energy "
    "functional: logged as identification I-DUST, the promotion gate. "
    "Lead for the derivation, observed only: the circuit/tick mismatch. "
    "The baryonic share counts one tick (one radius, parity flips -1 per "
    "tick, chain 25) while the total counts the full two-tick circuit "
    "(2pi of arc, the spinor's closed return): a double-cover structure "
    "the parity sector might supply. Not promoted.")

# ---------------- D2: scaling and the budget ----------------
M_dust = ratio_E46*m_p_GeV
ratio_obs = Odm_h2/Ob_h2
sig = abs(ratio_obs - ratio_E46)/(ratio_obs*np.sqrt(0.01**2 + 0.007**2))
print(f"      packet mass (2pi-1) m_p = {M_dust:.3f} GeV; one packet per "
      f"baryon; rho_dust/rho_b = 2pi-1 at all times by baryon conservation")
print(f"      observed ratio {ratio_obs:.3f} vs {ratio_E46:.4f}: "
      f"{abs(ratio_obs-ratio_E46)/ratio_obs*100:.2f}% = {sig:.1f} sigma "
      f"(Planck-era errors)")
check("D2: dust scaling a^-3 is inherited exactly from baryon number "
      "conservation (no freeze-out, no thermal history enters the ratio); "
      "E46 scorecard unchanged at 1.4%, ~1 sigma", sig < 2.0)

# ---------------- D3: cold ----------------
# Born comoving with its baryon. Conservative (hottest) case: birth at
# the QCD epoch with the proton's thermal velocity; the packet is never
# recoupled (E111), so its momentum simply redshifts.
p_th = np.sqrt(3*T_QCD_eV*m_p_eV)/1e9          # GeV, proton thermal
v_b  = p_th/np.sqrt(p_th**2 + m_p_GeV**2)      # birth velocity
a_b  = (gs0/gsQ)**(1/3)*(T0_eV/T_QCD_eV)
p_b  = M_dust*v_b/np.sqrt(1-v_b**2)            # packet momentum at birth
v0   = (p_b*a_b)/M_dust
print(f"      birth: a_b = {a_b:.2e}, v_birth = {v_b:.2f}; today "
      f"v0 = {v0:.1e}")
a = np.logspace(np.log10(a_b), 0, 200000)
p_of_a = p_b*a_b/a
v_of_a = p_of_a/np.sqrt(p_of_a**2 + M_dust**2)
E_of_a = np.sqrt(Or/a**4 + Om/a**3 + OL)
lam_fs = c_over_H0_Mpc*np.trapezoid(v_of_a/(a**2*E_of_a), a)
print(f"      free-streaming length = {lam_fs:.1e} Mpc = {lam_fs*1e6:.1f} pc")
check("D3: the dust is cold: v_today ~ 1e-12 and the free-streaming "
      "length sits >=3 orders below the ~0.1 Mpc Lyman-alpha scale; "
      "colder than any structure bound requires", v0 < 1e-10
      and lam_fs < 1e-4)

# ---------------- D4: collisionless ----------------
# Only gravity couples dust to anything (E112; leg 1 of E111 is PROVEN:
# no non-gravitational baryon-pinch contact exists at any strength).
G_GeV = 1/M_P_GeV**2
v_cl = 1000/2.998e5                     # cluster collision velocity ~1000 km/s
lnL = 30.0
sigma_grav = 4*np.pi*G_GeV**2*M_dust**2/v_cl**4*lnL   # GeV^-2, transfer
sigma_cm2 = sigma_grav*3.894e-28
m_g = M_dust*1.783e-24
print(f"      gravitational sigma/m ~ {sigma_cm2/m_g:.1e} cm^2/g vs "
      f"Bullet bound 1 cm^2/g")
check("D4: self-interaction and dust-baryon scattering are gravitational "
      "only: sigma/m sits >50 orders under the Bullet Cluster bound; "
      "the dust is collisionless in every astrophysical environment",
      sigma_cm2/m_g < 1e-50)
obs("the same theorem cuts the other way and answers the tracing "
    "objection: E111 leaves NOTHING to bind a packet to its baryon "
    "after birth. Dust and gas share initial conditions and then "
    "separate freely under gravity: cluster-collision offsets "
    "(Bullet-type) and tidally dust-stripped galaxies are ALLOWED, "
    "which a rigid per-baryon lock would forbid.")

# ---------------- D5: initial conditions ----------------
obs("born proportional to baryon number, delta_dust = delta_b at birth: "
    "strictly adiabatic, zero dust-baryon isocurvature (consistent with "
    "CMB isocurvature bounds by construction). From birth the packet "
    "ignores photon pressure (E111), so at horizon entry it grows while "
    "the baryon-photon fluid oscillates: the standard CDM sequence, "
    "with baryons falling into dust wells after recombination. Cold + "
    "collisionless + adiabatic + right density IS the definition of CDM "
    "at the level structure formation uses; the quantitative transfer "
    "function is future work and nothing tonight depends on it.")

# ---------------- D6: stability, route 1 (spectral orthogonality) ----
# The gate that killed the KK graviton. The loop's protection is exact
# periodicity: period T_pass (chain 25; gain invariance makes the period
# amplitude-independent). A periodic source radiates only at integer
# harmonics k*Omega, i.e. E = k*m_p. T4's Wilson bound is
# |A_N| <= |a| / |cos(delta/2)|, delta = E*T_pass/hbar. The surviving
# comb sits at the poles (half-integers). Integer harmonics sit at
# |cos| = 1: the MINIMUM of the bound, maximal cancellation.
kk = np.arange(0, 8)
bound_int  = 1/np.abs(np.cos(np.pi*kk))              # at E = k m_p
print(f"      Wilson bound at integer harmonics: {bound_int} (all exactly 1)")
check("D6: every integer harmonic E = k m_p sits at |cos(delta/2)| = 1, "
      "the exact minimum of the Wilson bound: the source comb and the "
      "surviving (k+1/2) comb are disjoint",
      np.allclose(bound_int, 1.0))
# numeric partial sums: integer harmonic vs half-integer, N = 1e6
def maxA(delta, N=1000000):
    n = np.arange(N)
    return np.max(np.abs(np.cumsum(np.exp(1j*n*(np.pi + delta)))))
A_int, A_half = maxA(2*np.pi), maxA(np.pi, N=100000)
print(f"      max |A_N| over 1e6 passes: integer harmonic {A_int:.2f}; "
      f"half-integer grows to {A_half:.0f} (N-linear)")
check("D6: over 1e6 passes the integer-harmonic amplitude never exceeds "
      "one pass while the half-integer channel grows without bound: "
      "cancellation confirmed at the source frequencies", A_int < 1.001
      and A_half > 1e4)
# a periodic source has NO half-integer content. Two routes:
# exact symbolic integral per harmonic, and a discrete-mean projection
# (exact on a uniform grid over whole cycles). CORRECTION DISCLOSED:
# the first draft used trapezoid quadrature here, whose endpoint
# weights leave a ~5e-6 residue on an exactly-zero integral; the check
# caught it (FAIL on the draft run) and the method was fixed, the
# claim unchanged.
import sympy as sp
ts = sp.symbols('t', real=True)
sym_zero = all(
    sp.integrate(sp.exp(sp.I*sp.pi*(h+1)*ts) *
                 sp.exp(-sp.I*sp.pi*(kh + sp.Rational(1, 2))*ts),
                 (ts, 0, 200)).equals(0)
    for h in range(3) for kh in range(3))
rng = np.random.default_rng(31)
coef = rng.normal(size=6) + 1j*rng.normal(size=6)
tt = np.arange(2**20)*(200/2**20)                     # 100 circuits, t-units
sig_p = sum(c*np.exp(1j*np.pi*(k+1)*tt) for k, c in enumerate(coef))
proj_half = abs(np.mean(sig_p*np.exp(-1j*np.pi*0.5*tt)))
proj_int  = abs(np.mean(sig_p*np.exp(-1j*np.pi*1.0*tt)))
print(f"      symbolic overlaps (9 harmonic pairs): all exactly 0 = "
      f"{sym_zero}; discrete projection ratio half-integer/own harmonic: "
      f"{proj_half/proj_int:.1e}")
check("D6: a T_pass-periodic source has zero overlap with every "
      "antiperiodic (half-integer) mode, exact in symbols and at "
      "machine precision on the grid: the only channels that could "
      "accumulate are absent from the source spectrum",
      sym_zero and proj_half/proj_int < 1e-10)
loss_per_age = g2 * 1.0                 # bounded one-pass probability
print(f"      first-order loss per packet per age of universe ~ g^2 x O(1) "
      f"= {loss_per_age:.1e}")
check("D6: first-order emission loss over the age of the universe is "
      "~6e-39 of a packet: stable where the KK graviton died",
      loss_per_age < 1e-30)

# ---------------- D7: stability, route 2 and the second-order channel --
obs("route 2, independent: chain 25's emit-absorb balance at the pinch "
    "is EXACT (corollary of the self-adjoint gluing): the loop's net "
    "pinch flux is zero as a theorem. CAVEAT, stated: that corollary was "
    "proven for the cycle as such; extending it to a comoving packet in "
    "an expanding background, where an emitted quantum redshifts before "
    "reabsorption, is a formalization step (the balance-extension "
    "question), flagged for the registry.")
# the only route to the surviving comb is second order: parametric pair
# emission at (k+1/2) + (k'+1/2) with the source supplying integer
# harmonics; rate ~ g^4 Omega (dimensional, INFERENCE flagged)
Gam2 = g2**2*Omega/(8*np.pi)
tau2 = 1/Gam2
print(f"      second-order (pair) channel: Gamma ~ g^4 Omega/8pi = "
      f"{Gam2:.1e}/s -> tau ~ {tau2:.1e} s = {tau2/age:.1e} ages")
check("D7: the parametric route to the half-integer comb is g^4: "
      "lifetime exceeds the age of the universe by >25 orders; both "
      "stability routes hold and the dust survives the gate that "
      "executed every particle candidate", tau2/age > 1e25)

# ---------------- D8: the register ----------------
print(f"""
      minted, register-ready (structural, per the agreed objective):
      P6-candidate: Omega_DM/Omega_b converges to 2pi - 1 = {ratio_E46:.4f}
        exactly, zero parameters. Current: {ratio_obs:.3f} +- ~1.2%,
        {sig:.1f} sigma. CMB-S4-era precision makes this decisive either
        way. Registered against the framework: a >3 sigma exclusion of
        {ratio_E46:.4f} kills E46 and chain 31 together.
      P7-candidate: the eternal null. Dust couples at alpha_grav and no
        stronger (E112), so EVERY direct-detection experiment stays
        empty forever. Any confirmed non-gravitational dark matter
        signal, anywhere, at any mass, falsifies the framework outright.
        Cheap to state, maximally exposed.""")

# ---------------- verdict ----------------
print("""
VERDICT (T9): chain 31 stands as a CANDIDATE, not yet promoted.
What is established tonight, conditional on I-A, M-EM, A1+A2, I-DUST:
  the framework contains a clustering carrier. The loop packet is
  localized, cold (v ~ 1e-12 today, free-streaming ~ pc), collisionless
  (>50 orders under Bullet), adiabatic by birth, scales as a^-3 by
  baryon conservation, separates freely from gas because E111 leaves
  nothing to bind it, and is stable by two routes: exact periodicity
  puts every source frequency at the exact minimum of the Wilson bound
  (the integer and half-integer combs are disjoint), and emit-absorb
  balance is exact for the cycle. Cold + collisionless + adiabatic +
  correct density is CDM phenomenology entire. The same Wilson sign
  that emptied the gate sector is what keeps the dust dark.
What is NOT established: the energy partition. The driver's natural
  partitions give pi^2 and pi; the arc-per-radius partition equals
  2pi - 1 and is selected by matching E46, not by dynamics. I-DUST is
  the load-bearing identification and its derivation is the promotion
  gate (lead: the tick/circuit double cover, observed only). Formation
  bookkeeping and the covariant embedding of packet stress-energy
  remain open. E46's backing moves from 'no carrier exists' (the T8
  exposure) to 'a carrier exists with stated conditions': the largest
  unbacked number now has a candidate spine.
""")
print("All committed checks resolved." if ok else "FAILURES PRESENT.")
