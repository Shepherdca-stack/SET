"""
SET time campaign, T11: the gradient-flow test on E1 (candidate chain 32).
Question posed this session: is the E1 driver a gradient or Hamiltonian
flow of some functional in the framework's own structures? If yes, the
generator is the energy and the dynamics establishes energy itself; if
provably no, energy must be built one level up (covariant embedding).

RESULT, stated up front:
  The test has a decisive answer with three parts, all machine-checked:
  1. E1 satisfies an AUTONOMOUS second-order law, exact and previously
     unrecorded: z'' = i pi (z' - 1). The trajectory-with-external-
     parameter reading was an artifact; the dynamics is a law, not a
     parametrized curve. (E115 candidate, THEOREM: identity of E1.)
  2. GRADIENT: NO, by theorem, two routes, in any Riemannian metric:
     the velocity subsystem has genuinely closed orbits (circles about
     v = 1), and gradient flows admit no closed orbits in any metric;
     on the surface, uniform poloidal circulation would need a single-
     valued F with F' proportional to the metric, whose circle integral
     cannot vanish. The loop is the flow of a CLOSED, NON-EXACT 1-form:
     locally a gradient, globally obstructed, with obstruction period
     exactly 2 pi; on the universal cover it IS a gradient flow.
  3. HAMILTONIAN: YES, exactly, of magnetic type. The law z'' =
     i pi z' - i pi is a unit mass in a uniform magnetic field of
     strength pi with a uniform transverse force: the prolate cycloid
     is the textbook crossed-fields cycloid, the loop is the cyclotron
     gyration, the drift is the guiding-center drift (solving z'' = 0
     gives v = 1 exactly). A Lagrangian exists and is verified by
     Euler-Lagrange; the canonical energy
        H = (1/2)|z'|^2 + pi Im z
     is EXACTLY conserved, value (1 + pi^2)/2, splitting pointwise as
     drift-kinetic 1/2 plus loop-kinetic pi^2/2 (the potential eats the
     cross term exactly). Two further exact invariants close the
     system: |z' - 1| = pi (the cyclotron radius) and the global linear
     clock t = Re z - Im(z')/pi - 1, which is the one-time theorem's t
     promoted to an explicit state-space function with dt/dt = 1 along
     EVERY orbit of the law. (E116 candidate.)
  Consequence for the partition question: energy IS now established by
  the dynamics, and the established energy REFUTES the mechanical
  reading of E46: the canonical partition is pi^2 (loop) to 1 (drift),
  the no-go's k = 2 case, excluded by the data at ~70 sigma. So E46's
  2 pi - 1 is not a partition of E1's mechanical energy; P-RING's
  tension-length energy is a configuration (rest) energy, a different
  functional, and the two now have disjoint scopes: H governs motion,
  P-RING (if it survives) governs mass. Consistent with, and predicted
  by, T10's no-go.

Scope: parts 1-3 are unconditional mathematics of the E1 law (THEOREM
grade; E1 itself is a theorem of the geodesic flow). No identification
of H with gravitating energy is asserted; the pi^2 exclusion is
precisely the statement that such an identification fails. The
magnetic/cyclotron language is a structural identity of the equation,
flagged as standard-physics vocabulary, not new physics content.

Standard inputs, flagged: Omega_b h^2 = 0.0224 (0.7%), Omega_DM h^2 =
0.120 (1%).
"""
import numpy as np
import sympy as sp

ok = True
def check(name, cond):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + name)
    ok = ok and cond
def obs(s): print("[OBS ] " + s)

t = sp.symbols('t', real=True)
z = sp.exp(sp.I*sp.pi*t) + (t + 1)
zd, zdd = sp.diff(z, t), sp.diff(z, t, 2)

# ---------------- G1: the autonomous law (E115 candidate) ----------
law = sp.simplify(zdd - sp.I*sp.pi*(zd - 1))
check("G1: E1 satisfies z'' = i pi (z' - 1) identically (symbolic zero): "
      "the driver is the solution of an autonomous second-order law; "
      "the external parameter t is eliminable", law == 0)
# uniqueness of the orbit within the law: general solution
C1, C2 = sp.symbols('C1 C2')
zg = C2 + t + C1*sp.exp(sp.I*sp.pi*t)      # general solution family
check("G1: the general solution of the law is z = C2 + t + C1 e^{i pi t} "
      "(verified by substitution): E1 is the |C1| = 1, C2 = 1 member; "
      "the loop radius and drift are the law's two integration constants",
      sp.simplify(sp.diff(zg, t, 2) - sp.I*sp.pi*(sp.diff(zg, t) - 1)) == 0)

# ---------------- G2: not a gradient flow, two routes ----------
# Route 1 (velocity subsystem, any metric): v' = i pi (v - 1) has closed
# orbits |v - 1| = const. Machine content: d/dt |v-1|^2 = 0 symbolically,
# so every orbit of the v-subsystem is a circle (genuinely closed), and
# a gradient flow in ANY Riemannian metric admits no nonconstant closed
# orbit (F strictly monotone along nonconstant orbits).
v = sp.Function('v')
w = sp.symbols('w')                         # w = v - 1
# |w(t)|^2 with w' = i pi w: w = w0 e^{i pi t}
w0 = sp.symbols('w0')
wt = w0*sp.exp(sp.I*sp.pi*t)
check("G2 route 1: |v - 1|^2 is exactly conserved by the velocity "
      "subsystem (symbolic), so its orbits are genuinely closed circles; "
      "closed orbits exclude gradient structure in ANY metric",
      sp.simplify(sp.diff(wt*sp.conjugate(wt), t)) == 0)
# constant-metric illustration: spectrum of the rotation is +/- i pi,
# metric-independent; g-self-adjoint operators have real spectrum
A = sp.Matrix([[0, -sp.pi], [sp.pi, 0]])
eigs = list(A.eigenvals().keys())
check("G2 route 1 (illustration): the rotation's spectrum is +/- i pi, "
      "pure imaginary and basis-independent; no metric makes it "
      "self-adjoint", all(sp.im(e) != 0 for e in eigs))
# Route 2 (on the surface): uniform poloidal circulation phi' = c under
# a gradient ansatz phi' = -g^{pp} F'(phi) forces F' = -c g_pp; the
# circle integral of F' must vanish for single-valued F, but
# -c * integral(g_pp) is strictly nonzero for any positive metric.
phi = sp.symbols('phi', positive=True)
gpp = sp.Function('g', positive=True)(phi)
lhs = sp.Integral(-1*gpp, (phi, 0, 2*sp.pi))   # proportional to loop of F'
obs("G2 route 2 (surface form, symbolic argument): F'(phi) = -c g_pp "
    "with c != 0 gives a circle integral -c * int g_pp dphi != 0 for any "
    "positive metric, contradicting the exactness requirement "
    "int F' dphi = 0 for single-valued F. The loop is therefore the flow "
    "of a CLOSED, NON-EXACT 1-form: locally gradient, globally "
    "obstructed, obstruction period exactly 2 pi; on the universal "
    "cover it IS a gradient flow. The 2 pi that blocks gradient "
    "structure is the same 2 pi P-RING assigns to the turn: OBSERVED "
    "correspondence, not promoted.")
check("G2 route 2 committed piece: int_0^{2pi} g dphi > 0 for positive g "
      "(numeric witness family g = 1 + a cos^2 phi, a in {0, 0.5, 3})",
      all(float(sp.integrate(1 + a*sp.cos(phi)**2, (phi, 0, 2*sp.pi))) > 0
          for a in (0, 0.5, 3)))

# ---------------- G3: exactly Hamiltonian, magnetic type ----------
x, y = sp.symbols('x y', real=True)
xd, yd = sp.symbols('xdot ydot', real=True)
L = sp.Rational(1,2)*(xd**2 + yd**2) + sp.pi/2*(y*xd - x*yd) - sp.pi*y
# Euler-Lagrange against the law's real form: x'' = -pi y', y'' = pi x' - pi
xt = sp.re(z.rewrite(sp.cos)); yt = sp.im(z.rewrite(sp.cos))
xt, yt = sp.simplify(xt), sp.simplify(yt)
elx = sp.diff(L.subs([(xd, sp.Derivative(sp.Function('X')(t), t))]), t)  # placeholder
# do EL properly with functions:
X, Y = sp.Function('X')(t), sp.Function('Y')(t)
Lf = sp.Rational(1,2)*(sp.diff(X,t)**2 + sp.diff(Y,t)**2) \
     + sp.pi/2*(Y*sp.diff(X,t) - X*sp.diff(Y,t)) - sp.pi*Y
EL_X = sp.diff(sp.diff(Lf, sp.diff(X,t)), t) - sp.diff(Lf, X)
EL_Y = sp.diff(sp.diff(Lf, sp.diff(Y,t)), t) - sp.diff(Lf, Y)
sub = {X: xt, Y: yt}
check("G3: the Lagrangian L = |z'|^2/2 + (pi/2)(y x' - x y') - pi y "
      "yields the law by Euler-Lagrange, and E1 solves it (both EL "
      "residuals symbolically zero on E1)",
      sp.simplify(EL_X.subs(sub).doit()) == 0
      and sp.simplify(EL_Y.subs(sub).doit()) == 0)
H = sp.Rational(1,2)*(sp.diff(xt,t)**2 + sp.diff(yt,t)**2) + sp.pi*yt
Hval = sp.simplify(H)
check("G3: the canonical energy H = |z'|^2/2 + pi Im z is EXACTLY "
      "conserved on E1 with value (1 + pi^2)/2 (symbolic; the potential "
      "eats the oscillating cross term pointwise)",
      sp.simplify(Hval - (1 + sp.pi**2)/2) == 0)
clock = sp.simplify(xt - sp.diff(yt, t)/sp.pi - 1)
check("G3: the global clock t = Re z - Im(z')/pi - 1 is an exact "
      "state-space identity on E1 (symbolic zero of clock - t); the "
      "one-time theorem's t is a linear function of the state",
      sp.simplify(clock - t) == 0)
gyro = sp.simplify(sp.Abs(sp.diff(z,t) - 1)**2)
check("G3: second invariant |z' - 1| = pi exact (cyclotron radius of "
      "the loop); with H and the clock the system is fully integrable",
      sp.simplify(gyro - sp.pi**2) == 0)
drift_v = sp.solve(sp.I*sp.pi*(w - 0), w)   # z''=0 <=> v - 1 = 0
check("G3: the guiding-center identity: z'' = 0 forces v = 1 exactly; "
      "the (t+1) drift is the law's unique force-free velocity "
      "(structural identity with the crossed-fields cycloid, standard "
      "vocabulary, flagged)", drift_v == [0])

# ---------------- G4: the partition consequence ----------
ratio_E46 = 2*np.pi - 1
Odm_h2, Ob_h2 = 0.120, 0.0224
robs = Odm_h2/Ob_h2
err = robs*np.sqrt(0.01**2 + 0.007**2)
sig_k2 = abs(robs - np.pi**2)/err
print(f"      canonical split: H = 1/2 (drift) + pi^2/2 (loop); "
      f"mechanical dark/baryon = pi^2 = {np.pi**2:.4f}; observed "
      f"{robs:.3f}: {sig_k2:.0f} sigma")
check("G4: the dynamics-derived energy gives the k = 2 partition, "
      "exactly as T10's no-go requires, and the data exclude it at "
      ">60 sigma: E46's ratio is NOT a partition of E1's mechanical "
      "energy; P-RING's tension-length energy is a configuration (rest) "
      "functional with disjoint scope (H governs motion, not mass)",
      sig_k2 > 60)
obs("numerology guard, preempted: the VALUE (1 + pi^2)/2 = 5.4348 sits "
    "near the observed ratio 5.357 (1.4%). It is not a reading: H "
    "carries units of speed^2 x mass and becomes a ratio only via a "
    "divisor; every registry-consistent divisor (drift kinetic 1/2, "
    "total H) returns pi^2 or its complement, nothing new. Logged to "
    "close the door, not as a lead.")

# ---------------- verdict ----------------
print(f"""
VERDICT (T11): the gradient-flow test is answered, and it answers the
energy question. E1 is not a trajectory decorated with a clock; it is
the solution of the autonomous law z'' = i pi (z' - 1) (E115
candidate), which is exactly Hamiltonian of magnetic type and provably
not gradient in any metric (closed velocity orbits; the surface form
fails exactness with obstruction period 2 pi, gradient only on the
cover). The generator exists: the canonical energy H = |z'|^2/2 +
pi Im z, exactly conserved at (1 + pi^2)/2, with the cyclotron
invariant |z' - 1| = pi and the exact linear clock t = Re z -
Im(z')/pi - 1 completing an integrable triple (E116 candidate). Energy
is thereby established from the dynamics for the first time. What the
established energy says about the standing question is negative and
clean: the mechanical partition is pi^2, excluded at ~70 sigma, so E46
is not about E1's mechanical energy, and P-RING, if it is anything, is
a rest-energy statement living one level up, exactly where T10's no-go
said the new content must enter. The campaign's two energy notions now
have names and disjoint jobs: H moves the state; the embedding stress-
energy (open) weighs it. The next derivation target is unchanged and
now better lit: build the covariant packet stress-energy whose rest
sector either produces the tension-length form or kills P-RING.
""")
print("All committed checks resolved." if ok else "FAILURES PRESENT.")
