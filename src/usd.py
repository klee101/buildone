"""Author a self-contained, meter-scale Isaac Sim prop from the accepted mesh cache."""
import json
import numpy as np

DEFAULT_DENSITY = 1000.0  # kg/m³, a nominal unmeasured solid density


def export(cache_path, path, static, mass):
    from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics, Vt

    with np.load(cache_path, allow_pickle=False) as cache:
        parts = json.loads(cache["metadata"].item())
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root)
        Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
        root.SetDocumentation(parts["scale_note"])
        root.SetCustomDataByKey("buildone:modelSha256", parts["model_sha256"])
        if not static:
            UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(root)
        if mass is None:
            attr = mass_api.CreateDensityAttr(DEFAULT_DENSITY)
            attr.SetDocumentation("Nominal BuildOne default density, in kg/m^3. Material density was unspecified.")
        else:
            mass_api.CreateMassAttr(mass)
        UsdGeom.Scope.Define(stage, "/Asset/Visual")
        guide = UsdGeom.Scope.Define(stage, "/Asset/Collision")
        guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

        def author_mesh(prim_path, key):
            pts, faces = cache[key+"_p"], cache[key+"_f"]
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([pts.min(axis=0), pts.max(axis=0)])))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            return mesh

        for name, part in parts["visual"].items():
            mesh = author_mesh("/Asset/Visual/"+name, part["mesh"])
            rgb = np.asarray(part["rgb"])
            linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*linear)])
        for name, key in parts["collision"].items():
            mesh = author_mesh("/Asset/Collision/"+name, key)
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        stage.GetRootLayer().Save()
    return dict(rigid_body=not static, mass_kg=mass, density_kg_m3=DEFAULT_DENSITY if mass is None else None,
                visual_parts=len(parts["visual"]), convex_colliders=len(parts["collision"]),
                origin="bottom center", meters_per_unit=1, up_axis="Z")
