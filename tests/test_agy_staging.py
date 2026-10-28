# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgySubstrate staging: suite plugins, leave-one-out arms, isolated HOME, side-effect shims.

Every test builds its own fake plugin `plugins/` checkout (incl. a `revmux` directory without
a plugin.json that must never be staged by the env fallback's `"*"` discovery) and its own fake
`$HOME`; the real `~/.gemini` is never touched. The suite comes from the deprecated env
fallback (SKILLBENCH_PLUGINS_DIR / SKILLBENCH_PLUGINS); suite.toml loading is covered by
tests/test_skillbench_suite.py.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.config import ConfigError
from harness2.core.spec import INSTALL_MODES, BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.agy import (
    AGY_SURFACE,
    INSTALL_MODE,
    PLUGINS_REL,
    SHIM_TOOLS,
    AgySubstrate,
    prepare_rollout,
)

AUTH = (".gemini/oauth_creds.json", ".gemini/google_accounts.json")
# What the env fallback discovers: every plugin directory holding a plugin.json.
SUITE_PLUGINS = ("brainstorm", "planning", "review", "skill-eval", "thinking-tools", "workflow")


def _write(p: Path, body: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _skill(plugin_dir: Path, name: str) -> None:
    _write(plugin_dir / "skills" / name / "SKILL.md", f"---\nname: {name}\n---\nDo {name}.\n")
    _write(plugin_dir / "skills" / name / "references" / "notes.md", f"{name} notes\n")


def _spec(tmp: Path) -> BenchSpec:
    ch = ChannelSpec("claude_cli", "m")
    return BenchSpec(
        name="skillbench",
        runs_root=tmp / "runs",
        install_mode=INSTALL_MODE,
        prompts=BenchPrompts(work="solve coding tasks"),
        channels={"proposer": ch, "selector": ch},
    )


def _tree_files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


class _StagingCase(unittest.TestCase):
    """Fake cc-thingz checkout + fake HOME + env pointing at them."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agy-staging-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        # Fake cc-thingz plugins/ checkout; `planning` is a symlink (like AGY's global
        # discovery links) and `review` holds a symlinked file, both must be resolved.
        self.plugins = self.tmp / "cc-thingz" / "plugins"
        _skill(self.plugins / "brainstorm", "brainstorm")
        real_planning = self.tmp / "elsewhere" / "planning"
        _skill(real_planning, "make-plan")
        _skill(real_planning, "exec")
        _write(real_planning / "plugin.json", '{"name": "cc-thingz-planning"}\n')
        _write(real_planning / "hooks.json", '{"hooks": {}}\n')
        self.plugins.mkdir(parents=True, exist_ok=True)
        (self.plugins / "planning").symlink_to(real_planning, target_is_directory=True)
        _skill(self.plugins / "review", "code-review-refactor")
        shared = _write(self.tmp / "shared" / "resolve-project-dir.sh", "#!/bin/sh\necho ok\n")
        (self.plugins / "review" / "scripts").mkdir(parents=True)
        (self.plugins / "review" / "scripts" / "resolve-project-dir.sh").symlink_to(shared)
        _write(self.plugins / "skill-eval" / "hooks.json", '{"hooks": {"PreInvocation": []}}\n')
        _skill(self.plugins / "thinking-tools", "root-cause-investigator")
        _skill(self.plugins / "workflow", "backlog")
        _write(self.plugins / "workflow" / "scripts" / "__pycache__" / "x.cpython-313.pyc")
        _skill(self.plugins / "revmux", "revmux")  # no plugin.json: never discovered
        for name in SUITE_PLUGINS:
            if not (self.plugins / name / "plugin.json").exists():
                _write(self.plugins / name / "plugin.json", f'{{"name": "fake-{name}"}}\n')

        # Fake real HOME: auth files plus things that must never leak into a rollout.
        self.home = self.tmp / "realhome"
        for rel in AUTH:
            _write(self.home / rel, f'{{"token": "{rel}"}}\n')
        _write(self.home / ".gemini" / "settings.json", '{"secret": true}\n')
        _write(self.home / ".gemini" / "GEMINI.md", "real user rules\n")
        _write(self.home / ".gemini" / "config" / "plugins" / "revmux" / "plugin.json", "{}\n")
        _write(self.home / ".ssh" / "id_rsa", "PRIVATE\n")

        self.tree = self.tmp / "tree"
        _write(self.tree / "GEMINI.md", "<!-- base harness -->\n")

        env = {
            "HOME": str(self.home),
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins),
            "SKILLBENCH_AUTH_FILES": ",".join(AUTH),
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SKILLBENCH_PLUGINS", None)

        self.sub = AgySubstrate()
        self.spec = _spec(self.tmp)

    def materialise(self, arm: str = "with", target: str = "make-plan", **extra):
        inst = self.sub.materialise(
            self.tree, self.spec, {"arm": arm, "target_skill": target, **extra}
        )
        self.addCleanup(shutil.rmtree, inst.staging_dir, True)
        return inst

    def staged_plugins(self, inst) -> Path:
        return inst.staging_dir / "home" / PLUGINS_REL


class TestPluginStaging(_StagingCase):
    def test_install_mode_registered(self) -> None:
        self.assertEqual(INSTALL_MODE, "agy_home")
        self.assertIn("agy_home", INSTALL_MODES)
        self.assertIs(self.sub.surface(), AGY_SURFACE)
        self.assertEqual(self.sub.name, "agy")

    def test_with_arm_stages_only_suite_plugins(self) -> None:
        inst = self.materialise("with")
        staged = self.staged_plugins(inst)
        self.assertEqual(
            sorted(p.name for p in staged.iterdir()),
            sorted(SUITE_PLUGINS),
        )
        self.assertFalse((staged / "revmux").exists())
        for skill in ("make-plan", "exec"):
            self.assertTrue((staged / "planning" / "skills" / skill / "SKILL.md").is_file())
        self.assertEqual(inst.extras["arm"], "with")
        self.assertEqual(inst.extras["target_skill"], "make-plan")
        self.assertEqual(inst.extras["install_mode"], "agy_home")
        self.assertEqual(inst.extras["suite"], "cc-thingz")  # parent dir of plugins/
        self.assertRegex(inst.extras["suite_hash"], r"^[0-9a-f]{16}$")

    def test_suite_hash_moves_with_the_checkout(self) -> None:
        before = self.materialise("with").extras["suite_hash"]
        _write(self.plugins / "review" / "skills" / "code-review-refactor" / "SKILL.md", "new\n")
        self.assertNotEqual(before, self.materialise("with").extras["suite_hash"])

    def test_symlinks_are_resolved(self) -> None:
        inst = self.materialise("with")
        staged = self.staged_plugins(inst)
        for p in [staged, *staged.rglob("*")]:
            self.assertFalse(p.is_symlink(), f"symlink staged: {p}")
        self.assertEqual(
            (staged / "review" / "scripts" / "resolve-project-dir.sh").read_text(),
            "#!/bin/sh\necho ok\n",
        )
        self.assertEqual(
            (staged / "planning" / "plugin.json").read_text(), '{"name": "cc-thingz-planning"}\n'
        )

    def test_pycache_not_staged(self) -> None:
        inst = self.materialise("with")
        self.assertEqual(list(self.staged_plugins(inst).rglob("__pycache__")), [])

    def test_without_arm_omits_only_target_skill(self) -> None:
        with_files = set(_tree_files(self.staged_plugins(self.materialise("with"))))
        inst = self.materialise("without", "make-plan")
        without_files = set(_tree_files(self.staged_plugins(inst)))
        dropped = with_files - without_files
        self.assertTrue(dropped)
        self.assertTrue(all(f.startswith("planning/skills/make-plan/") for f in dropped))
        self.assertEqual(without_files - with_files, set())
        staged = self.staged_plugins(inst)
        self.assertFalse((staged / "planning" / "skills" / "make-plan").exists())
        self.assertTrue((staged / "planning" / "skills" / "exec" / "SKILL.md").is_file())
        self.assertTrue((staged / "planning" / "hooks.json").is_file())
        self.assertTrue((staged / "skill-eval" / "hooks.json").is_file())

    def test_custom_allowlist(self) -> None:
        with mock.patch.dict(os.environ, {"SKILLBENCH_PLUGINS": "planning, review"}):
            inst = self.materialise("with")
        self.assertEqual(
            sorted(p.name for p in self.staged_plugins(inst).iterdir()), ["planning", "review"]
        )

    def test_missing_allowlisted_plugin_raises(self) -> None:
        with mock.patch.dict(os.environ, {"SKILLBENCH_PLUGINS": "planning,nope"}):
            with self.assertRaisesRegex(ConfigError, "nope"):
                self.materialise("with")

    def test_unknown_target_raises(self) -> None:
        for arm in ("with", "without"):
            with self.assertRaisesRegex(ValueError, "no-such-skill"):
                self.materialise(arm, "no-such-skill")

    def test_target_outside_allowlist_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "revmux"):
            self.materialise("without", "revmux")

    def test_bad_arm_or_missing_target_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "arm"):
            self.materialise("maybe")
        with self.assertRaisesRegex(ValueError, "target_skill"):
            self.sub.materialise(self.tree, self.spec, {"arm": "with"})
        with self.assertRaisesRegex(ValueError, "target_skill"):
            self.materialise("without", "../planning")

    def test_missing_plugins_dir_raises_with_remediation(self) -> None:
        os.environ.pop("SKILLBENCH_PLUGINS_DIR")
        with self.assertRaisesRegex(ConfigError, r"--suite.*SKILLBENCH_PLUGINS_DIR"):
            self.materialise("with")
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.tmp / "missing")
        with self.assertRaisesRegex(ConfigError, "SKILLBENCH_PLUGINS_DIR"):
            self.materialise("with")

    def test_wrong_install_mode_or_tree_raises(self) -> None:
        ch = ChannelSpec("claude_cli", "m")
        other = BenchSpec(
            name="x",
            runs_root=self.tmp,
            install_mode="codex_root",
            prompts=BenchPrompts(work="w"),
            channels={"proposer": ch, "selector": ch},
        )
        with self.assertRaisesRegex(ValueError, "agy_home"):
            self.sub.materialise(self.tree, other, {"arm": "with", "target_skill": "make-plan"})
        with self.assertRaisesRegex(ValueError, "agy harness tree"):
            self.sub.materialise(
                self.tmp / "empty", self.spec, {"arm": "with", "target_skill": "make-plan"}
            )

    def test_file_ctx_entries_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "file:"):
            self.materialise("with", **{"file:notes.md": "x"})

    def test_failed_materialise_leaves_no_staging_dir(self) -> None:
        before = set(Path(tempfile.gettempdir()).glob("adapt-agy-*"))
        with self.assertRaises(ValueError):
            self.materialise("with", "no-such-skill")
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("adapt-agy-*")), before)

    def test_base_tree_always_on_installed_as_global_rules(self) -> None:
        inst = self.materialise("with")
        self.assertEqual(
            (inst.staging_dir / "home" / ".gemini" / "GEMINI.md").read_text(),
            "<!-- base harness -->\n",
        )


class TestAuthAndHome(_StagingCase):
    def test_auth_files_copied_never_symlinked(self) -> None:
        inst = self.materialise("with")
        home = inst.staging_dir / "home"
        for rel in AUTH:
            p = home / rel
            self.assertTrue(p.is_file())
            self.assertFalse(p.is_symlink())
            self.assertEqual(p.read_text(), (self.home / rel).read_text())
        # writing the staged copy (a token refresh) never reaches the real HOME
        (home / AUTH[0]).write_text("refreshed\n")
        self.assertEqual((self.home / AUTH[0]).read_text(), f'{{"token": "{AUTH[0]}"}}\n')

    def test_nothing_else_from_real_home(self) -> None:
        inst = self.materialise("with")
        home = inst.staging_dir / "home"
        files = {
            str(p.relative_to(home))
            for p in home.rglob("*")
            if p.is_file() and PLUGINS_REL not in p.relative_to(home).parents
        }
        self.assertEqual(files, set(AUTH) | {".gemini/GEMINI.md"})
        self.assertNotEqual((home / ".gemini" / "GEMINI.md").read_text(), "real user rules\n")
        self.assertFalse((home / ".ssh").exists())
        self.assertFalse((home / ".gemini" / "settings.json").exists())

    def test_missing_auth_file_raises(self) -> None:
        os.environ["SKILLBENCH_AUTH_FILES"] = f"{AUTH[0]},.gemini/missing.json"
        with self.assertRaisesRegex(ConfigError, "missing.json"):
            self.materialise("with")

    def test_auth_files_optional(self) -> None:
        # internal Jetski authenticates without any file under $HOME: nothing is copied
        for unset in (None, "", " , "):
            if unset is None:
                os.environ.pop("SKILLBENCH_AUTH_FILES", None)
            else:
                os.environ["SKILLBENCH_AUTH_FILES"] = unset
            with self.subTest(value=unset):
                inst = self.materialise("with")
                home = inst.staging_dir / "home"
                files = {
                    str(p.relative_to(home))
                    for p in home.rglob("*")
                    if p.is_file() and PLUGINS_REL not in p.relative_to(home).parents
                }
                self.assertEqual(files, {".gemini/GEMINI.md"})

    def test_auth_files_must_stay_in_home(self) -> None:
        for bad in ("../etc/passwd", "/etc/passwd"):
            os.environ["SKILLBENCH_AUTH_FILES"] = bad
            with self.assertRaisesRegex(ConfigError, "SKILLBENCH_AUTH_FILES"):
                self.materialise("with")

    def test_real_home_unchanged(self) -> None:
        before = _tree_files(self.home)
        inst = self.materialise("without")
        prepare_rollout(inst, self.tmp / "run1", base_env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(_tree_files(self.home), before)

    def test_fingerprint_stable_and_differs_between_arms(self) -> None:
        a = self.materialise("with")
        b = self.materialise("with")
        self.assertEqual(a.fingerprint, b.fingerprint)
        # auth content (e.g. a refreshed token) does not move the harness identity
        (self.home / AUTH[0]).write_text("rotated\n")
        self.assertEqual(self.materialise("with").fingerprint, a.fingerprint)
        w = self.materialise("without")
        self.assertNotEqual(a.fingerprint, w.fingerprint)
        self.assertEqual(w.fingerprint, self.materialise("without").fingerprint)
        self.assertNotEqual(
            w.fingerprint, self.materialise("without", "code-review-refactor").fingerprint
        )
        _write(self.tree / "GEMINI.md", "changed\n")
        self.assertNotEqual(self.materialise("with").fingerprint, a.fingerprint)


class TestShimsAndRolloutEnv(_StagingCase):
    def test_shims_present_executable_and_identical_in_both_arms(self) -> None:
        w = self.materialise("with")
        wo = self.materialise("without")
        self.assertEqual(
            sorted(p.name for p in (w.staging_dir / "shims").iterdir()), sorted(SHIM_TOOLS)
        )
        for tool in SHIM_TOOLS:
            p = w.staging_dir / "shims" / tool
            self.assertTrue(os.access(p, os.X_OK), tool)
        self.assertEqual(
            _tree_files(w.staging_dir / "shims"), _tree_files(wo.staging_dir / "shims")
        )

    def test_prepare_rollout_env(self) -> None:
        base = {"PATH": "/usr/bin:/bin", "EDITOR": "vim", "XDG_CONFIG_HOME": str(self.home)}
        envs = {}
        for arm in ("with", "without"):
            inst = self.materialise(arm)
            run = self.tmp / f"run-{arm}"
            env = prepare_rollout(inst, run, base_env=base)
            envs[arm] = (env, run, inst)
            self.assertEqual(env["HOME"], str(run / "home"))
            self.assertEqual(env["EDITOR"], "true")
            self.assertEqual(env["VISUAL"], "true")
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(inst.staging_dir / "shims"))
            self.assertTrue(env["PATH"].endswith("/usr/bin:/bin"))
            self.assertEqual(env["SKILLBENCH_SIDE_EFFECTS_DIR"], str(run / "side_effects"))
            self.assertTrue(env["XDG_CONFIG_HOME"].startswith(str(run / "home")))
            self.assertTrue((run / "side_effects").is_dir())
            self.assertTrue((run / "home" / PLUGINS_REL / "planning").is_dir())
            for rel in AUTH:
                self.assertFalse((run / "home" / rel).is_symlink())
                self.assertTrue((run / "home" / rel).is_file())
        self.assertEqual(base["EDITOR"], "vim")  # base_env not mutated

        # identical across arms apart from the per-run paths
        def norm(env, run, inst):
            out = {}
            for k, v in env.items():
                out[k] = v.replace(str(run), "<run>").replace(str(inst.staging_dir), "<stg>")
            return out

        self.assertEqual(norm(*envs["with"]), norm(*envs["without"]))

    def test_prepare_rollout_fresh_home_per_run(self) -> None:
        inst = self.materialise("with")
        run = self.tmp / "run"
        prepare_rollout(inst, run, base_env={"PATH": "/bin"})
        _write(run / "home" / "leftover.txt")
        (run / "home" / AUTH[0]).write_text("refreshed\n")
        prepare_rollout(inst, run, base_env={"PATH": "/bin"})
        self.assertFalse((run / "home" / "leftover.txt").exists())
        self.assertEqual((run / "home" / AUTH[0]).read_text(), f'{{"token": "{AUTH[0]}"}}\n')
        # the staged template is untouched by rollouts
        self.assertEqual(
            (inst.staging_dir / "home" / AUTH[0]).read_text(), f'{{"token": "{AUTH[0]}"}}\n'
        )

    def test_shims_write_stdin_to_side_effects(self) -> None:
        inst = self.materialise("with")
        run = self.tmp / "run"
        env = prepare_rollout(inst, run, base_env={"PATH": os.environ.get("PATH", os.defpath)})
        for tool in SHIM_TOOLS:
            r = subprocess.run(
                [tool, "-selection", "clipboard"] if tool == "xclip" else [tool],
                input=f"copied by {tool}\n",
                text=True,
                env=env,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(
                (run / "side_effects" / f"{tool}.txt").read_text(), f"copied by {tool}\n"
            )
        # a second write appends instead of clobbering
        subprocess.run(["pbcopy"], input="again\n", text=True, env=env, timeout=10, check=True)
        self.assertEqual(
            (run / "side_effects" / "pbcopy.txt").read_text(), "copied by pbcopy\nagain\n"
        )

    def test_shim_output_mode_reads_nothing(self) -> None:
        inst = self.materialise("with")
        run = self.tmp / "run"
        env = prepare_rollout(inst, run, base_env={"PATH": os.environ.get("PATH", os.defpath)})
        for argv in (["xclip", "-o"], ["xsel", "--output"], ["xsel", "-o", "-b"]):
            r = subprocess.run(
                argv, input="ignored\n", text=True, env=env, capture_output=True, timeout=10
            )
            self.assertEqual(r.returncode, 0)
            self.assertEqual(r.stdout, "")
        self.assertEqual(list((run / "side_effects").iterdir()), [])

    def test_shim_without_side_effects_dir_discards(self) -> None:
        inst = self.materialise("with")
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(self.tmp / "h")}
        r = subprocess.run(
            [str(inst.staging_dir / "shims" / "wl-copy")],
            input="lost\n",
            text=True,
            env=env,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(r.returncode, 0)
        self.assertFalse((self.tmp / "h").exists())


if __name__ == "__main__":
    unittest.main()
