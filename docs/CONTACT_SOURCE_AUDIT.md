# Bounded original-contact source audit

Source: Reid Y. Kawamoto, *The Avatar Paradigm in Granular Materials*, Caltech
doctoral thesis (2018), [author repository PDF](https://thesis.caltech.edu/11042/6/RYKawamoto-2018-thesis.pdf).
Accessed 2026-10-31. Receipt:
`evidence/kawamoto-thesis-contact-audit-v1/receipt.json`. SHA-256:
`c60c79155d2b08aa0fb6fbc72143f4c41571fab920cbd4bf9e355ea671076099`.
The 58,822,286-byte PDF remains private; its copyright page reserves all rights.
This is the thesis version, not a verified exact copy of the 2016 journal paper.

Rendered and inspected PDF pages 51–53 and 56 (printed 39–41 and 44). Equations
3.9–3.12 give a linear normal nodal spring, opposite reaction and common-node
moment arms. Equations 3.13–3.20 describe relative velocity, shear history and
Coulomb limiting. Equation 3.24 places global linear velocity damping in the
translational motion equation; equations 3.25–3.27 address rotation. The wall text
on printed page 44 uses the grain stiffness for walls.

These inspected sections do not establish the restitution implementation used by
the later Feldfogel benchmark. They do not specify an area-weighted restitution
conversion, effective mass per surface node, or the later wall quadrature weights.
Do not transfer an ordinary spring/dashpot coefficient into a traction law without
deriving its units and surface-resolution dependence. The current exponential
global drag and the verified project plane operator therefore remain distinct from
an exact reproduction of the benchmark's restitution input.

Next: obtain the benchmark-specific damping/weighting evidence, or explicitly
define and independently qualify a different project damping variant. Either path
must retain the unresolved source identity; neither completes the published
reproduction gate by itself. See WALL_CONTACT.md and PUBLISHED_ASSEMBLY_REFERENCE.md.

A later primary-source thesis audit and independent oscillator/tributary-area
analysis are now recorded in CONTACT_DAMPING_DESIGN.md. They identify an explicit
body-mass convention and ambiguous printed logarithm placement in that distinct
source, but do not resolve the Feldfogel benchmark's area-weighted damping choice.
The new dimensional audit demonstrates why a nodal coefficient cannot be inferred
from its restitution number. No normal dashpot has been added to the Engine.
