"""CPU coverage for current requests, mixed-client semantics and preset isolation."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import qwen35_e1pd4_mixed_video_perf_smoke as preset
import qwen35_e1pd4_perf_smoke as original
import qwen35_epd_fusion_4gpu_smoke as smoke
from qwen35_mixed_video import load_workload, output_token_budget, summarize

DATA = Path(__file__).with_name("qwen35_mixed_video_data")
EXPECTED = dict(prompt_tokens=100, video_tokens=80)


def response_body():
    return dict(
        choices=[dict(index=0, message=dict(content="answer"), finish_reason="stop")],
        usage=dict(
            prompt_tokens=100,
            completion_tokens=3,
            prompt_tokens_details=dict(video_tokens=80),
        ),
        aux_info=dict(pd_sep=False, reuse_len=0),
    )


def send(body=None, status=200, payload=None):
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.text = json.dumps(response_body() if body is None else body)
    if status != 200:
        response.raise_for_status.side_effect = RuntimeError("HTTP error")
    with mock.patch("requests.post", return_value=response):
        return smoke.request(
            "http://test/v1/chat/completions",
            payload or dict(stream=False),
            0,
            EXPECTED,
        )


def parse(module=preset, *args):
    with mock.patch.object(smoke, "run_launcher", side_effect=lambda a: a):
        return module.main(["--output=/tmp/mixed-preset-test", *args])


class MixedVideoTest(unittest.TestCase):
    def test_unified_gate_environment_reaches_worker(self):
        for value in ("0", "1"):
            with mock.patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": value}):
                a = parse()
                command = smoke.bazel_command(a)
                self.assertIn("--test_env=RTP_BF16_GATE_KERNEL=" + value, command)
                forwarded = [x[len("--test_arg="):] for x in command
                             if x.startswith("--test_arg=")]
                with mock.patch.object(smoke, "run_worker", side_effect=lambda a: a), \
                     mock.patch.object(preset, "validate_runtime"):
                    worker = preset.main(["--worker", *forwarded])
                self.assertEqual(
                    {k: v for k, v in worker.preset_metadata["environment"].items()
                     if k.startswith("RTP_BF16_")},
                    {"RTP_BF16_GATE_KERNEL": value},
                )

    def test_prompt_contracts_reject_gibberish_despite_stop(self):
        from qwen35_mixed_video import validate_content

        valid = {
            1: "# Core Summary\nx\n# Scene\nx\n# Subjects Description\nx\n# Dynamic Narrative\nx\n# Style, Color\nx",
            2: '{"scenes": [{"start": null, "end": null, "description": "A court."}], "visual_events": [], "on_screen_text": []}',
            3: "# Core Summary\nx\n# Visible Subjects\nx\n# Action and Interaction Timeline\nx\n# On-screen Text\nx\n# Visibility Limitations\nx",
        }
        valid[4] = valid[1]
        for fixture, answer in valid.items():
            self.assertEqual(validate_content("reasoning</think>" + answer, fixture), [])
            self.assertTrue(validate_content("ER sp  % gibberish", fixture))
        self.assertEqual(validate_content("```json\n" + valid[2] + "\n```", 2), [])
        for invalid in (
            '{"ambience_events": [], "sound_events": []}',
            '{"scenes": "bad", "visual_events": [], "on_screen_text": []}',
            '{"scenes": ["A court"], "visual_events": [], "on_screen_text": []}',
            '{"scenes": [{"start": 0, "end": null, "description": "A court"}], "visual_events": [], "on_screen_text": []}',
            '{"scenes": [], "visual_events": [], "on_screen_text": [{"start": null, "end": null, "text": "EXIT"}]}',
        ):
            self.assertTrue(validate_content(invalid, 2), invalid)
        self.assertTrue(validate_content("Speaker 1\nAppearance: visible.\nLanguage: English.", 3))
        self.assertTrue(validate_content(("A whole repeated line with enough characters.\n" * 10) + valid[3], 3))
        self.assertTrue(validate_content(("again " * 32) + valid[3], 3))
        # Long, varied reasoning is not rejected just because it exceeds8192.
        long_reasoning = " ".join("word" + str(i) for i in range(9000))
        self.assertEqual(validate_content(long_reasoning + "</think>" + valid[3], 3), [])
        body = response_body()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.text = json.dumps(body)
        with mock.patch("requests.post", return_value=response):
            row = smoke.request("http://test", dict(stream=False), 0, dict(EXPECTED, fixture_id=3))
        self.assertEqual(row["finish_reason"], "stop")
        self.assertFalse(row["ok"])
        self.assertIn("missing requested visual subject sections", row["validation_errors"])

    def test_repeated_short_line_blocks_fail_even_when_answer_stops(self):
        from qwen35_mixed_video import validate_content

        # Each timestamp line is shorter than the single-line repetition filter.
        block = "\n".join(f"`<00:0{i}.0 second>`" for i in range(4)) + "\n"
        answer = '{"scenes": [], "visual_events": [], "on_screen_text": []}'
        self.assertEqual(validate_content(block * 2 + "</think>" + answer, 2), [])
        errors = validate_content(block * 4 + "</think>" + answer, 2)
        self.assertIn(
            "degenerate generation: multi-line block repeated at least 4 times",
            errors,
        )

    def test_final_answer_does_not_inflate_reasoning_repetition(self):
        from qwen35_mixed_video import validate_content

        answer = (
            '{\n'
            '  "scenes": [{"start": "00:00.0", "end": "00:03.0", "description": "An indoor basketball court."}],\n'
            '  "visual_events": [{"start": "00:02.0", "end": "00:03.0", "description": "A basketball passes through a net."}],\n'
            '  "on_screen_text": []\n'
            '}\n'
        )
        draft = "```json\n" + answer + "```\n"
        reasoning = "".join(draft + f"Review distinct timing detail {i}.\n" for i in range(3))
        self.assertEqual(validate_content(reasoning + "</think>" + draft, 2), [])
        # The threshold within either section stays unchanged.
        self.assertIn(
            "degenerate generation: multi-line block repeated at least 4 times",
            validate_content(reasoning + draft + "</think>" + draft, 2),
        )

    def test_current_request_preserved_except_video_location(self):
        workload, manifest = load_workload(DATA)
        for entry, source in zip(workload, manifest["requests"]):
            actual = copy.deepcopy(entry["payload"])
            for msg in actual["messages"]:
                for part in msg["content"]:
                    if part["type"] == "video_url":
                        self.assertTrue(Path(part["video_url"]["url"]).is_file())
                        self.assertEqual(
                            part["preprocess_config"],
                            dict(fps=6, min_frames=4, max_frames=90),
                        )
                        part["video_url"]["url"] = source["video_url"]
            self.assertEqual(
                actual, json.loads((DATA / source["template"]).read_text())
            )

    def test_omitted_output_limit_preserves_other_request_fields_and_sources(self):
        original, manifest = load_workload(DATA)
        uncapped, _ = load_workload(DATA, omit_max_tokens=True)
        for before, after in zip(original, uncapped):
            expected = copy.deepcopy(before)
            del expected["payload"]["max_tokens"]
            self.assertEqual(after, expected)
        for entry in manifest["requests"]:
            self.assertEqual(
                json.loads((DATA / entry["template"]).read_text())["max_tokens"], 8192
            )
        self.assertEqual(output_token_budget({}, 24599, 65536, 32000), 32000)
        self.assertEqual(output_token_budget({}, 40642, 65536, 32000), 24894)
        self.assertEqual(
            output_token_budget({"max_tokens": 8192}, 40642, 65536, 32000), 8192
        )
        with self.assertRaises(ValueError):
            output_token_budget({}, 65536, 65536, 32000)
        with self.assertRaises(ValueError):
            output_token_budget({"max_tokens": 32000}, 40642, 65536, 32000)

    def test_omit_max_tokens_forwarding_and_default_isolation(self):
        a = parse(preset, "--omit-max-tokens", "--mega-moe-chunk-tokens=0")
        command = smoke.bazel_command(a)
        forwarded = [
            x[len("--test_arg=") :] for x in command if x.startswith("--test_arg=")
        ]
        self.assertIn("--omit-max-tokens", forwarded)
        with mock.patch.object(
            smoke, "run_worker", side_effect=lambda a: a
        ), mock.patch.object(preset, "validate_runtime"):
            b = preset.main(["--worker", *forwarded])
        self.assertTrue(b.omit_max_tokens)
        self.assertIsNone(b.preset_metadata["max_tokens"])
        self.assertEqual(b.mega_moe_chunk_tokens, 0)
        self.assertFalse(parse().omit_max_tokens)
        self.assertEqual(parse().preset_metadata["max_tokens"], 8192)
        self.assertFalse(hasattr(parse(original), "omit_max_tokens"))

    def test_concurrency_forwarding_keeps_server_and_original_preset_unchanged(self):
        for level in ("4", "8", "16", "32", "64", "128", "256", "512", "8,32,128,512"):
            with self.subTest(concurrency=level):
                a = parse(
                    preset,
                    "--steady-concurrency=" + level,
                    "--omit-max-tokens",
                    "--mega-moe-chunk-tokens=0",
                )
                command = smoke.bazel_command(a)
                forwarded = [
                    x[len("--test_arg=") :]
                    for x in command
                    if x.startswith("--test_arg=")
                ]
                with mock.patch.object(
                    smoke, "run_worker", side_effect=lambda a: a
                ), mock.patch.object(preset, "validate_runtime"):
                    b = preset.main(["--worker", *forwarded])
                self.assertEqual(b.steady_concurrency, level)
                self.assertEqual(
                    b.preset_metadata["settings"]["steady_concurrency"], level
                )
                self.assertEqual(b.native_fp8_attn, 1)
                self.assertEqual(b.fp8_kv_cache, 1)
                self.assertEqual(b.seq_size_per_block, 4096)
                self.assertNotIn(
                    "RTP_TRTLLM_DECODE_BATCH_INVARIANT",
                    b.preset_metadata["environment"],
                )
                self.assertFalse(
                    any(
                        option.startswith("--test_env=RTP_TRTLLM_DECODE_BATCH_INVARIANT=")
                        for option in command + b.bazel_option
                    )
                )
                self.assertIn("--test_env=RTP_BF16_GATE_KERNEL=1", b.bazel_option)
                self.assertEqual(b.preset_metadata["environment"]["RTP_BF16_GATE_KERNEL"], "1")
                self.assertEqual(
                    {k: v for k, v in b.preset_metadata["environment"].items()
                     if k.startswith("RTP_BF16_")},
                    {"RTP_BF16_GATE_KERNEL": "1"},
                )
                self.assertEqual(
                    [x for x in b.bazel_option if x.startswith("--test_env=RTP_BF16_")],
                    ["--test_env=RTP_BF16_GATE_KERNEL=1"],
                )
                self.assertEqual(b.decode_graph, 1)
                self.assertEqual(b.rank_concurrency, 256)
                self.assertEqual(b.client_processes, 4)
                self.assertEqual(b.steady_warmup_seconds, 240)
                self.assertEqual(b.steady_window_seconds, 300)
                self.assertEqual(b.steady_windows, 2)
                self.assertTrue(b.omit_max_tokens)
                self.assertEqual(b.mega_moe_chunk_tokens, 0)
        self.assertEqual(parse().steady_concurrency, "512")
        self.assertEqual(parse(original).steady_concurrency, "512")
        self.assertEqual(original.DEFAULTS["steady_concurrency"], "512")
        self.assertEqual(parse(original).native_fp8_attn, 1)
        with self.assertRaises(SystemExit):
            parse(preset, "--native-fp8-attn=0")
        with self.assertRaises(SystemExit):
            parse(original, "--steady-concurrency=256")
        with self.assertRaises(SystemExit):
            parse(preset, "--steady-concurrency=24")

    def test_runfiles_symlink_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for source in DATA.rglob("*"):
                target = root / source.relative_to(DATA)
                if source.is_dir():
                    target.mkdir(exist_ok=True)
                else:
                    target.symlink_to(source.resolve())
            self.assertEqual(len(load_workload(root)[0]), 4)

    def test_corrupted_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(DATA, Path(d) / "data")
            (Path(d) / "data/videos/video_04.mp4").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_workload(Path(d) / "data")

    def test_nonstream_response_and_complete_evidence(self):
        row = send()
        self.assertTrue(row["ok"], row)
        self.assertEqual(json.loads(row["raw_response"]), response_body())
        self.assertNotIn("ttft_s", row)
        self.assertNotIn("tpot_ms", row)

    def test_invalid_responses_fail(self):
        for label in ("empty", "length", "tokens", "pd", "cache", "choices"):
            body = response_body()
            if label == "empty":
                body["choices"][0]["message"]["content"] = ""
            if label == "length":
                body["choices"][0]["finish_reason"] = "length"
            if label == "tokens":
                body["usage"]["prompt_tokens"] = 101
            if label == "pd":
                body["aux_info"]["pd_sep"] = True
            if label == "cache":
                body["aux_info"]["reuse_len"] = 1
            if label == "choices":
                body["choices"] = []
            with self.subTest(label=label):
                self.assertFalse(send(body)["ok"])
        row = send(dict(error="overloaded"), status=503)
        self.assertFalse(row["ok"])
        self.assertIn("overloaded", row["raw_response"])
        with mock.patch("requests.post", side_effect=TimeoutError("timeout")):
            self.assertFalse(
                smoke.request("http://test", dict(stream=False), 0, EXPECTED)["ok"]
            )

    def test_all_four_requests_select_matching_expectations(self):
        workload, _ = load_workload(DATA)
        expected = {"by_fixture": {str(i): dict(EXPECTED) for i in range(1, 5)}}
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.text = json.dumps(response_body())
        rng = mock.MagicMock()
        rng.choice.side_effect = workload
        with mock.patch.object(
            smoke.random, "SystemRandom", return_value=rng
        ), mock.patch("requests.post", return_value=response) as post:
            rows = [
                smoke.request("http://test", workload, i, expected) for i in range(4)
            ]
        self.assertEqual([r["fixture_id"] for r in rows], [1, 2, 3, 4])
        self.assertTrue(all(r["ok"] for r in rows))
        self.assertEqual(
            [c.kwargs["json"] for c in post.call_args_list],
            [e["payload"] for e in workload],
        )

    def test_preset_and_forwarding(self):
        a = parse()
        self.assertTrue(a.mixed_video)
        self.assertEqual(a.max_input_tokens, 65536)
        self.assertEqual(a.steady_concurrency, "512")
        command = smoke.bazel_command(a)
        self.assertIn(preset.TARGET, command)
        forwarded = [
            x[len("--test_arg=") :] for x in command if x.startswith("--test_arg=")
        ]
        with mock.patch.object(
            smoke, "run_worker", side_effect=lambda a: a
        ), mock.patch.object(preset, "validate_runtime"):
            b = preset.main(["--worker", *forwarded])
        for name in (
            "mixed_video",
            "max_input_tokens",
            "data_dir",
            "client_processes",
            "steady_concurrency",
        ):
            self.assertEqual(getattr(a, name), getattr(b, name))
        old = parse(original)
        self.assertFalse(old.mixed_video)
        self.assertEqual(old.max_input_tokens, 32768)
        _, args = smoke.server_config("4,5,6,7", max_input_tokens=65536)
        for field in (
            "max_seq_len",
            "max_batch_tokens_size",
            "max_batch_tokens_without_cache",
        ):
            self.assertIn(f"--{field} 65536", args)
        with self.assertRaises(SystemExit):
            parse(preset, "--max-input-tokens=32768")

    def test_diagnostics_require_preflight_and_forward_to_worker(self):
        with self.assertRaises(SystemExit):
            parse(preset, "--grid-sync-diag")
        a = parse(preset, "--preflight-only", "--grid-sync-diag")
        command = smoke.bazel_command(a)
        self.assertIn("--test_env=DG_MEGA_MOE_FP8_GRID_SYNC_DIAG=1", command)
        self.assertIn("--test_arg=--preflight-only", command)
        self.assertIn("--test_arg=--grid-sync-diag", command)
        forwarded = [
            x[len("--test_arg=") :] for x in command if x.startswith("--test_arg=")
        ]
        with mock.patch.object(
            smoke, "run_worker", side_effect=lambda a: a
        ), mock.patch.object(preset, "validate_runtime"):
            b = preset.main(["--worker", *forwarded])
        self.assertTrue(b.preflight_only)
        self.assertEqual(
            b.preset_metadata["environment"]["DG_MEGA_MOE_FP8_GRID_SYNC_DIAG"], "1"
        )
        self.assertEqual(
            parse().preset_metadata["environment"]["DG_MEGA_MOE_FP8_GRID_SYNC_DIAG"],
            "0",
        )

    def test_chunk_disabled_by_default_reaches_worker_and_preserves_reference(self):
        a = parse()
        command = smoke.bazel_command(a)
        self.assertIn("--test_env=RTP_MEGAMOE_CHUNK_TOKENS=0", command)
        forwarded = [
            x[len("--test_arg=") :] for x in command if x.startswith("--test_arg=")
        ]
        with mock.patch.object(
            smoke, "run_worker", side_effect=lambda a: a
        ), mock.patch.object(preset, "validate_runtime"):
            b = preset.main(["--worker", *forwarded])
        self.assertEqual(b.mega_moe_chunk_tokens, 0)
        self.assertEqual(
            b.preset_metadata["environment"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0"
        )
        self.assertEqual(
            parse().preset_metadata["environment"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0"
        )
        self.assertNotIn(
            "RTP_MEGAMOE_CHUNK_TOKENS", parse(original).preset_metadata["environment"]
        )
        self.assertFalse(b.preflight_only)
        self.assertFalse(b.grid_sync_diag)
        self.assertEqual(b.steady_concurrency, "512")

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "not-created"
            with mock.patch.object(
                preset, "validate_runtime"
            ) as runtime, mock.patch.object(smoke.subprocess, "run") as run:
                self.assertEqual(
                    preset.main(["--output=" + str(out), "--data-dir=/missing"]), 0
                )
            self.assertFalse(out.exists())
            runtime.assert_not_called()
            run.assert_not_called()

    def test_window_accounting_by_fixture(self):
        rows = [
            dict(
                fixture_id=i,
                started_s=230,
                completed_s=250,
                client_rank=i - 1,
                ok=True,
                e2e_s=20,
                usage=dict(completion_tokens=3),
            )
            for i in range(1, 5)
        ]
        rows += [dict(rows[0], completed_s=900)]
        result = summarize(rows, 240, 300, 2, smoke.steady_summary)
        self.assertEqual(result["1"]["measurement"]["successful"], 1)
        self.assertEqual(result["1"]["drain_completions"], 1)
        self.assertEqual(result["4"]["rounds"][0]["successful"], 1)
        self.assertNotIn("ttft_s", result["1"]["measurement"])


if __name__ == "__main__":
    unittest.main()
