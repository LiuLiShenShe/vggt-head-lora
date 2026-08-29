"""P4-1: four-path v4 verdict must record uses_test_reference_pose=true."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_four_path_provenance():
    p = os.path.join(ROOT, "four_path_v4", "metric_definitions.json")
    assert os.path.exists(p), "four_path_v4/metric_definitions.json 不存在"
    d = json.load(open(p))
    flat = json.dumps(d)
    assert "uses_test_reference_pose" in flat, "must declare uses_test_reference_pose"
    assert "uses_test_reference_point_geometry" in flat, "must distinguish pose from point geometry"
    assert "evaluation_only" in flat, "must declare evaluation_only"
    # Verify reference pose is used but point geometry is not (check nested provenance if present)
    prov = d.get("provenance_v32", d)  # may be nested under provenance_v32
    assert prov.get("uses_test_reference_pose") is True, f"uses_test_reference_pose must be True, got {prov}"
    assert prov.get("uses_test_reference_point_geometry") is False, f"reference point geometry must be False"


if __name__ == "__main__":
    test_four_path_provenance()
    print("ALL four-path provenance tests passed")
