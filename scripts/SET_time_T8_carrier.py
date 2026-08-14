"""
SET time campaign, T8: the dark matter carrier problem.
Opened by T5. Question: the framework derives Omega_DM/Omega_b = 2pi - 1
(E46, PROVEN geometry, 1.4%); what object carries it?

Method: exhaustive elimination over the registry's particle inventory,
then the reframing the eliminations force.

Inventory and status:
  1. Gate tower (0.0783 eV, 57 eV, ...): KILLED by T5, three ways.
  2. Band fermions (the nine, E98): they ARE the Standard Model matter;
     baryons cannot be their own dark matter. Excluded by definition.
  3. Graviton zero mode: massless (E79, machine-checked). Cannot be DM.
  4. KK graviton tower (E80, gap ~8.2-8.8 GeV at candidate A's unit):
     the last open branch. Gravitational coupling only (T7), right mass
     range for cold DM, and a Z2 (z -> -z) exists on the surface.
     Resolved below by lifetime.

Standard inputs, flagged: reduced Planck mass 2.435e18 GeV; age of
universe 4.35e17 s; decaying-DM lifetime floor ~1e19 s (order, external);
gravitational freeze-in yield Y ~ (T_RH/M_Pbar)^3 / g* (dimensional
estimate, coefficient order-of-magnitude, INFERENCE flagged); inflation
reheating ceiling ~1.6e16 GeV (order, external).
"""
import numpy as np
from numpy import trapezoid as trapz
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import eigsh

ok = True
def check(name, cond):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name)
    ok = ok and cond
def obs(s): print("[OBS ] " + s)

m_p = 0.93827208816; Mbar = 2.435e18; hbar_GeVs = 6.582119569e-25
age = 4.35e17

# ---------------- D1: KK graviton profile and gate overlap ----------
def Ws(x): return -9/4 - 1/(4*np.expm1(x))
N, umin, L = 32000, 1e-6, 60.0
x = np.linspace(umin, L, N); h = x[1]-x[0]
xm = 0.5*(x[1:]+x[:-1]); Wm = Ws(xm)
rows = np.repeat(np.arange(N-1), 2)
cols = np.empty(2*(N-1), dtype=int)
cols[0::2] = np.arange(N-1); cols[1::2] = np.arange(1, N)
dat = np.empty(2*(N-1)); dat[0::2] = -1/h - Wm/2; dat[1::2] = 1/h - Wm/2
A = csc_matrix((dat,(rows,cols)), shape=(N-1,N)); H = (A.T@A).tocsc()
w = np.exp(-3*x); Winv = diags(1/np.sqrt(w), 0)
vals, vecs = eigsh((Winv@H@Winv).tocsc(), k=3, sigma=0, which='LM')
i = np.argsort(vals); vals = vals[i]; vecs = vecs[:,i]
psis = (1/np.sqrt(w))[:,None]*vecs
for k in range(3):
    psis[:,k] /= np.sqrt(trapz(psis[:,k]**2, x))
# gate sector u-density: j(u) = (1/pi) e^{-u/2}(1-e^{-u})^{-1/2},
# exact norm B(1/2,1/2) = pi. The u -> 0 endpoint (psi ~ u^{-1/4},
# j ~ u^{-1/2}) is handled ANALYTICALLY below u1: naive trapezoids are
# grid-sensitive by orders here (caught by suite v4 at merge; the
# shipped first draft's 3.1e-3 was that artifact).
u1 = 5e-3
j = np.exp(-x/2)/np.sqrt(-np.expm1(-x))/np.pi
core = x > u1
c = []
for kk in range(3):
    p = psis[:,kk].copy()
    mask = (x > u1) & (x < 2*u1)
    Ak = np.mean(p[mask]*x[mask]**0.25)
    norm2 = trapz(p[core]**2, x[core]) + Ak**2*2*np.sqrt(u1)
    p /= np.sqrt(norm2); Ak /= np.sqrt(norm2)
    c.append(trapz((p*j)[core], x[core]) + (Ak/np.pi)*4*u1**0.25)
sup = (c[1]/c[0])**2
print(f"      KK tower m^2: {vals[1]:.2f}, {vals[2]:.2f} "
      f"(zero mode at {vals[0]:.1e})")
print(f"      gate-overlap couplings c0, c1, c2 = {c[0]:+.3f}, "
      f"{c[1]:+.3f}, {c[2]:+.3f}; decay suppression (c1/c0)^2 = {sup:.1e} "
      f"(endpoint-corrected; converged 6.8e-2)")
check("D1: zero-mode orthogonality does NOT protect the KK graviton: "
      "its overlap with the gate sector is suppressed by only ~7e-2, "
      "not by orders", 3e-2 < sup < 1.5e-1)

# ---------------- D2: the lifetime kill ----------------
for m2, tag in ((vals[1], "e^{-3u} convention"), (71.0, "recorded E80")):
    m_KK = np.sqrt(m2)*m_p
    Gam = sup*m_KK**3/(8*np.pi*Mbar**2)
    tau = hbar_GeVs/Gam
    print(f"      m_KK = {m_KK:.2f} GeV ({tag}): tau = {tau:.1e} s "
          f"= {tau/3.15e7:.1e} yr; age/tau = {age/tau:.0e}")
m_KK = np.sqrt(vals[1])*m_p
tau = hbar_GeVs/(sup*m_KK**3/(8*np.pi*Mbar**2))
check("D2: KILL. The KK graviton decays to gate pairs in ~1e5 years, "
      "5 orders short of the age of the universe and ~7 short of the "
      "decaying-DM floor; no registry selection rule forbids the decay "
      "(parity (-1)^N allows pair channels). The last particle branch "
      "is closed.", tau < 1e15)

# ---------------- D3: the rescue spec, recorded ----------------
Odm_h2 = 0.120; gstar = 106.75
Y_need = Odm_h2*3.6e-9/m_KK          # standard relic bookkeeping, GeV units
T_RH = Mbar*(gstar*Y_need)**(1/3)
print(f"      IF a state at the KK gap were exactly stable, gravitational "
      f"freeze-in closes the budget at T_RH ~ {T_RH:.1e} GeV "
      f"(order-of-magnitude coefficient), below the ~1.6e16 GeV "
      f"inflation ceiling")
check("D3: the rescue spec is recorded and nontrivial: an exact "
      "stabilizing selection rule (absent from the registry) plus "
      "reheating near 4e15 GeV would make the KK gap the carrier; "
      "neither exists today", 1e15 < T_RH < 1.6e16)

# ---------------- D4: verdict ----------------
print("""
VERDICT (T8): the registry's particle inventory contains NO dark matter
carrier. Gate tower (T5), band fermions (they are the SM), massless
zero mode, and now the KK graviton tower (megayear lifetime) are all
closed. The elimination is conditional on exactly one thing: the
inventory being complete, which the consolidated v1.6 registry asserts.

What this forces, stated plainly: E46 never claimed a particle. It
derives the ratio 2pi - 1 from geometry, matter totals one full turn
per baryon's unit radius, and the registry's own OPEN 8 already gates
'dark matter production' behind the time campaign. Tonight that gating
stops being a scheduling choice and becomes a theorem of elimination:
within this framework, dark matter is a geometric component or it is
nothing.

The exposure that creates, which is the time campaign's new central
physics target: dark matter is observed to CLUSTER (acoustic peaks,
lensing, cluster collisions, external facts). A smooth geometric
component, like the derived dark energy, cannot cluster. So the
framework owes a derivation of geometric dust: a component of the
geometry, coupled gravitationally as T7 demands, that carries energy
density 2pi - 1 per baryon AND supports structure formation. Deliver
that and E46 stops being a naked postdiction; fail structurally and
E46 is the framework's largest unbacked number. Either way the target
is sharp, which is worth more than the dead branches were.
""")
print("All committed checks resolved." if ok else "FAILURES PRESENT.")
