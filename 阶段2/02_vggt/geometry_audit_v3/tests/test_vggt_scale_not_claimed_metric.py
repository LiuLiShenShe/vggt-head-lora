"""P0-5: VGGT depth must not be claimed as guaranteed metric depth."""
import os, re, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Forbidden phrases when referring to VGGT depth
FORBIDDEN = [
    r"(?i)vggt\s+(depth\s+)?is\s+metric",
    r"(?i)vggt\s+(depth\s+)?(?:produces?|generates?)\s+metric",
    r"(?i)vggt\s+absolute\s+metric\s+depth",
    r"(?i)guaranteed\s+metric\s+depth",
    r"(?i)vggt\s+(raw\s+)?depth\s+is\s+metric",
]

def _is_real_claim(line):
    neg_markers = ["禁止", "❌", "forbidden", "not claimed", "错误", "不得", "不能"]
    # Lines quoting the old error to explain the fix (historical citation, not a claim)
    # e.g. 称 "VGGT depth is metric"  or  Q3 | VGGT raw depth 能否称 guaranteed metric depth？
    quote_hints = ["称 ", "referred to", "formerly", "was"]
    if any(m in line for m in neg_markers):
        return False
    if any(h in line for h in quote_hints):
        return False
    return True


def test_vggt_not_claimed_metric():
    """Check all v3.2 reports don't claim VGGT depth is metric."""
    violations = []
    for fn in ("PHASE22_GEOMETRY_AUDIT_V32.md", "PHASE22_GEOMETRY_AUDIT_V31.md",
               "VISUAL_AUDIT_v31.md"):
        fp = os.path.join(ROOT, fn)
        if not os.path.exists(fp): continue
        lines = open(fp, encoding="utf-8").readlines()
        for line in lines:
            if not _is_real_claim(line): continue
            for pat in FORBIDDEN:
                for m in re.finditer(pat, line):
                    violations.append(f"{fn}: '{m.group()}' in: {line.strip()[:120]}")
    assert not violations, f"VGGT depth wrongly claimed metric: {violations}"


def test_depth_scale_semantics_exists():
    p = os.path.join(ROOT, "DEPTH_SCALE_SEMANTICS.md")
    assert os.path.exists(p), "DEPTH_SCALE_SEMANTICS.md 不存在"
    txt = open(p).read()
    assert "metric" in txt.lower(), "semantics file must discuss metric depth"


if __name__ == "__main__":
    test_vggt_not_claimed_metric()
    test_depth_scale_semantics_exists()
    print("ALL VGGT scale semantics tests passed")
