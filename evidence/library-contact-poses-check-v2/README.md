# Analytical fixture assertion history

The v1 fixture test passes the cube but fails an anisotropic box because its
componentwise comparison imposes an absolute 1e-25 m^3 threshold on analytically
zero transverse traction. FP64 triangle projection produces transverse cancellation
of about 6.26e-25 m^3, while the axial traction is about 3.38e-11 m^3. The original
assertion and failed log are retained in library-contact-poses-check-v1.

The v2 fixture compares the traction vector's relative norm error at the same
1e-7 relative tolerance and independently checks the analytical separation and
both active-node counts. Both boxes pass. This corrects the fixture's comparison
metric; it does not alter the separately frozen contact-pose acceptance policy,
source geometry, source samples, physical poses in earlier studies, 0.1% GPU
force gate, 10% quadrature gate, 2% refinement gate or 3% overlap guard. The full
library study is not scientifically admitted by these software fixtures.
