"""brain.update.install — diff + pip-from-ensurepip + smoke + swap (#286 §3.3).
No network: pip runs with --no-index --find-links <tmp>."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from brain.update import install, overlay


def _wheel(dirpath: Path, dist: str, version: str, files: dict[str, str]) -> Path:
    """Write a minimal installable wheel; return its path."""
    tag = dist.replace("-", "_")
    whl = dirpath / f"{tag}-{version}-py3-none-any.whl"
    info = f"{tag}-{version}.dist-info"
    meta = {
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    allf = {**files, **meta}
    with zipfile.ZipFile(whl, "w") as z:
        for name, body in allf.items():
            z.writestr(name, body)
        z.writestr(f"{info}/RECORD", "".join(f"{n},,\n" for n in allf) + f"{info}/RECORD,,\n")
    return whl


def _pin(whl: Path, name: str, version: str) -> str:
    sha = hashlib.sha256(whl.read_bytes()).hexdigest()
    return f"{name}=={version} \\\n    --hash=sha256:{sha}\n"


def _fake_bundle(tmp_path: Path, installed: dict[str, str], bundle_id: str = "bundle-1") -> Path:
    site = tmp_path / "bundle-site"
    site.mkdir()
    for name, version in installed.items():
        info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                       encoding="utf-8")
    (site / "_ce_bundle_id").write_text(bundle_id + "\n", encoding="utf-8")
    return site


BRAIN_OK = {
    "brain/__init__.py": "",
    "brain/cli.py": "",
    "brain/bridge/__init__.py": "",
    "brain/bridge/server.py": "",
    "brain/chat/__init__.py": "",
    "brain/chat/engine.py": "",
}


def test_parse_requirements_splits_options_and_pins():
    text = ("--index-url https://pypi.org/simple\n"
            "--extra-index-url https://download.pytorch.org/whl/cpu\n"
            "Certifi==2026.4.22 \\\n    --hash=sha256:aa \\\n    --hash=sha256:bb\n"
            "torch==2.14.0+cpu ; sys_platform != 'darwin' \\\n    --hash=sha256:cc\n")
    options, reqs = install.parse_requirements(text)
    assert options == ["--index-url https://pypi.org/simple",
                       "--extra-index-url https://download.pytorch.org/whl/cpu"]
    assert [(r.name, r.version) for r in reqs] == [("certifi", "2026.4.22"), ("torch", "2.14.0+cpu")]
    assert "--hash=sha256:bb" in reqs[0].block and "sys_platform" in reqs[1].block


def test_parse_requirements_accepts_pep508_extras():
    text = "requests[socks]==2.31.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    options, reqs = install.parse_requirements(text)
    assert options == []
    assert [(r.name, r.version) for r in reqs] == [("requests", "2.31.0")]
    assert "--hash=sha256:" in reqs[0].block


def test_parse_requirements_rejects_editable_lines():
    with pytest.raises(install.UpdateError, match="unrecognised requirement line"):
        install.parse_requirements("-e .\n")


def test_diff_keeps_only_changed_pins_and_never_package_managers():
    _, reqs = install.parse_requirements(
        "certifi==2026.4.22 \\\n    --hash=sha256:aa\n"
        "idna==3.10 \\\n    --hash=sha256:bb\n"
        "setuptools==84.0.0 \\\n    --hash=sha256:cc\n")
    todo = install.diff_requirements(reqs, {"certifi": "2026.4.22", "idna": "3.9"})
    assert [r.name for r in todo] == ["idna"]


def test_installed_versions_reads_only_the_given_site_dir(tmp_path):
    site = _fake_bundle(tmp_path, {"Certifi": "2026.4.22", "idna": "3.9"})
    assert install.installed_versions(site) == {"certifi": "2026.4.22", "idna": "3.9"}


def test_apply_installs_only_the_diff_and_the_brain_then_activates(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    changed = _wheel(finds, "demo-changed", "2.0", {"demo_changed/__init__.py": ""})
    same = _wheel(finds, "demo-same", "1.0", {"demo_same/__init__.py": ""})
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text(_pin(changed, "demo-changed", "2.0") + _pin(same, "demo-same", "1.0"), encoding="utf-8")
    site = _fake_bundle(tmp_path, {"demo-changed": "1.0", "demo-same": "1.0"})
    root = tmp_path / "brain-overlay"
    entry = install.apply_update(wheel=brain_whl, requirements=req, commit="c" * 40, site_dir=site,
                                 root=root, pip_extra=["--no-index", "--find-links", str(finds)])
    folder = root / entry["dir"]
    names = sorted(p.name for p in folder.iterdir())
    assert "demo_changed" in names and "brain" in names and "demo_same" not in names
    assert overlay.read_state(root)["active"] == entry
    assert entry["brain_version"] == "9.9.9" and entry["bundle_id"] == "bundle-1"


def test_failed_smoke_leaves_the_active_overlay_untouched(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    broken = _wheel(finds, "companion-emergence", "9.9.9", {**BRAIN_OK, "brain/cli.py": "raise ImportError('boom')\n"})
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    overlay.activate(root, {"dir": "old", "commit": "o" * 40, "brain_version": "0.0.42", "bundle_id": "bundle-1"})
    before = (root / "current.json").read_bytes()
    with pytest.raises(install.UpdateError, match="does not load"):
        install.apply_update(wheel=broken, requirements=req, commit="d" * 40, site_dir=site, root=root,
                             pip_extra=["--no-index", "--find-links", str(finds)])
    assert (root / "current.json").read_bytes() == before
    assert not [p for p in root.iterdir() if p.name.startswith(".staging-")]


def test_hash_mismatch_fails_before_activation(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    dep = _wheel(finds, "demo-changed", "2.0", {"demo_changed/__init__.py": ""})
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("demo-changed==2.0 \\\n    --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
    site = _fake_bundle(tmp_path, {"demo-changed": "1.0"})
    root = tmp_path / "brain-overlay"
    with pytest.raises(install.UpdateError, match="pip failed"):
        install.apply_update(wheel=brain_whl, requirements=req, commit="e" * 40, site_dir=site, root=root,
                             pip_extra=["--no-index", "--find-links", str(finds)])
    assert overlay.read_state(root)["active"] is None
    assert dep.exists()


def test_a_bundle_without_the_hook_is_refused(tmp_path):
    site = tmp_path / "old-bundle-site"
    site.mkdir()
    with pytest.raises(install.UpdateError, match="predates"):
        install.apply_update(wheel=tmp_path / "x.whl", requirements=tmp_path / "r.txt", commit="f" * 40,
                             site_dir=site, root=tmp_path / "brain-overlay")


def test_reapplying_the_same_commit_reuses_the_folder(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}
    entry = install.apply_update(**kw)
    marker = root / entry["dir"] / "marker"
    marker.write_text("kept", encoding="utf-8")
    overlay.revert(root)
    install.apply_update(**kw)
    assert marker.exists() and overlay.read_state(root)["active"]["commit"] == "a" * 40


def test_failed_swap_keeps_the_active_overlay(tmp_path, monkeypatch):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}
    entry = install.apply_update(**kw)
    folder = root / entry["dir"]
    # corrupt the stamp so a re-apply of the SAME commit + bundle targets this same folder
    (folder / "stamp.json").write_text("not json", encoding="utf-8")

    real_replace = install.os.replace

    def fake_replace(src, dst, *a, **kw2):
        if Path(src).name.startswith(".staging-"):
            raise OSError("boom")
        return real_replace(src, dst, *a, **kw2)

    monkeypatch.setattr(install.os, "replace", fake_replace)

    with pytest.raises((install.UpdateError, OSError)):
        install.apply_update(**kw)

    active = overlay.read_state(root)["active"]
    assert active == entry
    assert (root / active["dir"]).is_dir()
    assert (root / active["dir"] / "brain").is_dir()


def test_a_named_folder_with_a_bad_stamp_is_never_touched(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}
    entry1 = install.apply_update(**kw)
    old_folder = root / entry1["dir"]
    (old_folder / "stamp.json").write_text("not json", encoding="utf-8")
    (old_folder / "sentinel").write_text("keep-me", encoding="utf-8")

    entry2 = install.apply_update(**kw)

    assert entry2["dir"] != entry1["dir"]
    assert old_folder.is_dir()
    assert (old_folder / "sentinel").read_text(encoding="utf-8") == "keep-me"
    assert (root / entry2["dir"]).is_dir()
    state = overlay.read_state(root)
    assert state["active"] == entry2
    assert state["previous"] == entry1


def test_fresh_name_never_reuses_a_named_folder(tmp_path, monkeypatch):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}

    entry1 = install.apply_update(**kw)
    canonical = root / entry1["dir"]
    (canonical / "stamp.json").write_text("not json", encoding="utf-8")
    (canonical / "sentinel").write_text("keep-canonical", encoding="utf-8")

    entry2 = install.apply_update(**kw)
    r1_folder = root / entry2["dir"]
    (r1_folder / "stamp.json").write_text("not json", encoding="utf-8")
    (r1_folder / "sentinel").write_text("keep-r1", encoding="utf-8")

    # prune keeps only active+previous (a single-generation history, by existing
    # design, unrelated to this fix) — disable it so this test can isolate what the
    # SWAP itself does to the two folders that are named at the moment of the third
    # apply, instead of conflating that with the separately-tested prune behaviour.
    monkeypatch.setattr(install.overlay, "prune", lambda root: None)

    entry3 = install.apply_update(**kw)

    assert entry3["dir"] == f"{entry1['dir']}-r2"
    assert entry3["dir"] not in (entry1["dir"], entry2["dir"])
    assert canonical.is_dir() and (canonical / "sentinel").read_text(encoding="utf-8") == "keep-canonical"
    assert r1_folder.is_dir() and (r1_folder / "sentinel").read_text(encoding="utf-8") == "keep-r1"
    assert (root / entry3["dir"]).is_dir()
    state = overlay.read_state(root)
    assert state["active"] == entry3
    assert state["previous"] == entry2


def test_new_bundle_id_installs_into_a_new_folder(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    (tmp_path / "s1").mkdir()
    (tmp_path / "s2").mkdir()
    site1 = _fake_bundle(tmp_path / "s1", {}, bundle_id="bundle-1")
    site2 = _fake_bundle(tmp_path / "s2", {}, bundle_id="bundle-2")
    root = tmp_path / "brain-overlay"
    entry1 = install.apply_update(wheel=brain_whl, requirements=req, commit="b" * 40, site_dir=site1,
                                  root=root, pip_extra=["--no-index", "--find-links", str(finds)])
    entry2 = install.apply_update(wheel=brain_whl, requirements=req, commit="b" * 40, site_dir=site2,
                                  root=root, pip_extra=["--no-index", "--find-links", str(finds)])
    assert entry1["dir"] != entry2["dir"]
    assert (root / entry1["dir"]).is_dir()
    assert (root / entry2["dir"]).is_dir()
    assert overlay.read_state(root)["active"] == entry2
    assert overlay.read_state(root)["previous"] == entry1


def test_leftover_unnamed_folder_is_replaced(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    leftover = root / f"{'c' * 12}-{'bundle-1'[:8]}"
    leftover.mkdir(parents=True)
    (leftover / "junk").write_text("garbage", encoding="utf-8")

    entry = install.apply_update(wheel=brain_whl, requirements=req, commit="c" * 40, site_dir=site,
                                 root=root, pip_extra=["--no-index", "--find-links", str(finds)])

    folder = root / entry["dir"]
    assert folder == leftover
    assert (folder / "brain").is_dir()
    assert not (folder / "junk").exists()
    stamp = json.loads((folder / "stamp.json").read_text(encoding="utf-8"))
    assert stamp["commit"] == "c" * 40 and stamp["bundle_id"] == "bundle-1"


def test_stamp_only_unnamed_folder_is_reinstalled_not_reused(tmp_path):
    """A half-pruned folder (e.g. Windows .pyd locked during prune's rmtree) can keep
    a matching stamp.json with no brain/ under it. Reuse must require the folder be
    named by current.json — else a re-apply activates a folder with no brain
    (final-review finding 1)."""
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}
    base = f"{kw['commit'][:12]}-bundle-1"
    target = root / base
    target.mkdir(parents=True)
    stamp = {"dir": base, "commit": kw["commit"], "brain_version": "9.9.9", "bundle_id": "bundle-1"}
    (target / "stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    # no brain/ under target: simulates a half-pruned folder that kept only its stamp

    entry = install.apply_update(**kw)

    assert entry["dir"] == base
    assert (target / "brain").is_dir()  # reinstalled, not silently reused as-is
    assert overlay.read_state(root)["active"] == entry


def test_redirected_name_that_exists_on_disk_unnamed_is_replaced(tmp_path):
    """Deferred T3: when the canonical name is already named by current.json and the
    counter picks an `-r<n>` name that happens to already exist on disk but isn't
    itself named (a stray leftover), the leftover is replaced, not reused."""
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    kw = {"wheel": brain_whl, "requirements": req, "commit": "a" * 40, "site_dir": site, "root": root,
          "pip_extra": ["--no-index", "--find-links", str(finds)]}
    entry1 = install.apply_update(**kw)  # names `base` as active
    canonical = root / entry1["dir"]
    (canonical / "stamp.json").write_text("not json", encoding="utf-8")  # forces a redirect

    stray = root / f"{entry1['dir']}-r1"
    stray.mkdir(parents=True)
    (stray / "junk").write_text("garbage", encoding="utf-8")

    entry2 = install.apply_update(**kw)  # base is named -> redirects to the -r1 stray on disk

    assert entry2["dir"] == f"{entry1['dir']}-r1"
    assert (stray / "brain").is_dir()
    assert not (stray / "junk").exists()


def test_stamp_carries_installed_at_but_current_json_entry_does_not(tmp_path):
    """Spec gap: stamp.json gets an installed_at timestamp (diagnostic only); the
    entry stored in current.json stays exactly {dir, commit, brain_version,
    bundle_id} so it round-trips through equality checks unchanged."""
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    req = tmp_path / "requirements.txt"
    req.write_text("", encoding="utf-8")
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    entry = install.apply_update(wheel=brain_whl, requirements=req, commit="a" * 40, site_dir=site,
                                 root=root, pip_extra=["--no-index", "--find-links", str(finds)])
    stamp = json.loads((root / entry["dir"] / "stamp.json").read_text(encoding="utf-8"))
    assert "installed_at" in stamp and stamp["installed_at"].endswith("Z")
    assert set(entry) == {"dir", "commit", "brain_version", "bundle_id"}
    assert overlay.read_state(root)["active"] == entry


def test_unsafe_commit_labels_are_refused(tmp_path):
    for bad in ("../x", "/abs", "", ".hidden", "a/b"):
        with pytest.raises(install.UpdateError, match="unsafe commit"):
            install.apply_update(wheel=tmp_path / "x.whl", requirements=tmp_path / "r.txt", commit=bad,
                                 site_dir=tmp_path, root=tmp_path / "brain-overlay")


def test_missing_requirements_file_is_an_update_error(tmp_path):
    finds = tmp_path / "finds"
    finds.mkdir()
    brain_whl = _wheel(finds, "companion-emergence", "9.9.9", BRAIN_OK)
    site = _fake_bundle(tmp_path, {})
    root = tmp_path / "brain-overlay"
    with pytest.raises(install.UpdateError, match="update failed"):
        install.apply_update(wheel=brain_whl, requirements=tmp_path / "missing.txt", commit="a" * 40,
                             site_dir=site, root=root, pip_extra=["--no-index", "--find-links", str(finds)])
    assert overlay.read_state(root)["active"] is None
    assert not [p for p in root.iterdir() if p.name.startswith(".staging-")]
