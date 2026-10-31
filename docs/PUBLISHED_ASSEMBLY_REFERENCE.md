# Published assembly reference acquisition

Access attempt: 2026-10-31. Status: source identified; raw data not acquired;
no assembly reproduction or experimental validation claimed.

[Feldfogel et al., IJNME 125(5), e7400 (2024)](https://doi.org/10.1002/nme.7400)
is the contact-method source already identified in CONTACT_METHOD.md. Its readable
publisher HTML, section 3.2 / Figure 4, specifies 27 ordered spheres, fixed side
and bottom boundaries, and an upper moving wall. Section 3.3 / Figure 5 instead
uses interlocked blocks and a spherical indenter. These are distinct benchmarks.
The paper directs readers to ETH for supporting data.

The primary ETH search listing identifies
[the associated dataset](https://doi.org/10.3929/ethz-b-000640908), dated 2023-11-08,
with a ZIP and PDF readme under CC BY 4.0. The landing page returned HTTP 403 through
the web reader; the public item API returned HTTP 429 through curl. No bytes were
downloaded, so archive checksums, file inventory, source units and completeness
remain unverified. Do not infer numerical inputs from figure appearance or indexed
metadata. Retry acquisition after the rate limit clears; inspect the readme and
data before freezing any reproduction protocol.

## Implementation implications

The current laboratory implements periodic orthogonal cells. It has neither an
explicit wall contact API nor an independently qualified wall-reaction measurement.
A periodic-cell approximation would therefore be a different study and cannot
complete the published-benchmark gate. Source acquisition must establish dimensions,
initial placement/orientations, loading schedule, contact parameters, surface/grid
discretizations, damping and output units. Missing items remain gaps.

The smallest next implementation is a generic prescribed plane-boundary contact
contract with reaction measurement, followed by independent force/work verification;
only then can an exact source-supported wall-confined benchmark be frozen. Keep
the project's two directed half-weight passes explicit when comparing with the
publication. No benchmark-specific handwritten dynamics or fitted acceptance curve
is authorized by this source note. Physical sand validation remains a separate gate.
