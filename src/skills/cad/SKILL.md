---
name: cad
description: Model a single photographed object as editable build123d solids and convex collision pieces, then review one orthographic build preview.
---

Use the original source crop(s) for observed geometry. The isolated crop identifies the selected instance among neighbors. The four generated views can suggest hidden structure. When they disagree with a visible surface, follow the photograph.

Write an importable Python model using build123d. Work in millimeters with X across, Y deep and Z up. Front, left and right cameras are at -Y, -X and +X. In the top view (+Z), the front edge is at the bottom. Honor the identified physical front. Group named size parameters at the top. Match all supplied axes and estimate any missing dimensions. Define SCALE_NOTE as a short string explaining the scale evidence.

Return (visuals, collisions) from build(). Each is a dictionary keyed by distinct descriptive Python identifiers within its group. A visual value is (build123d_shape, (r, g, b)) with sRGB floats from 0 to 1. A collision value is either a convex build123d shape or the string name of a convex visual to reuse. The exporter cooks each collider as one convex hull. Split rings, bridges and support gaps into a modest compound so apertures and contact surfaces remain usable. Keep the source self-contained. Use primitives, booleans, profiles and extrusions for silhouette, holes, grips and bearing surfaces.

The runner executes the model and returns measured dimensions, a CPU preview or the concrete build error. In the same session, accept a usable first build with KEEP. Make at most one targeted revision when the evidence shows a material silhouette, dimension, support, grip, clearance or execution problem. The runner performs all builds.
