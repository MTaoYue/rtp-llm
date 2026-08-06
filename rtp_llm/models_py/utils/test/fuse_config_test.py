import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.models_py.utils.fuse_config import (
    fuse_kernels_enabled,
    glm5_prefill_refine_enabled,
)


class FuseConfigTest(TestCase):
    def test_defaults_enable_master_and_refine_paths(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_FUSE_KERNELS", None)
            os.environ.pop("RTP_LLM_GLM5_PREFILL_REFINE", None)
            self.assertTrue(fuse_kernels_enabled())
            self.assertTrue(glm5_prefill_refine_enabled())

    def test_refine_switch_preserves_master_switch(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_FUSE_KERNELS": "1",
                "RTP_LLM_GLM5_PREFILL_REFINE": "0",
            },
            clear=False,
        ):
            self.assertTrue(fuse_kernels_enabled())
            self.assertFalse(glm5_prefill_refine_enabled())

    def test_master_switch_disables_refine_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_FUSE_KERNELS": "0",
                "RTP_LLM_GLM5_PREFILL_REFINE": "1",
            },
            clear=False,
        ):
            self.assertFalse(fuse_kernels_enabled())
            self.assertFalse(glm5_prefill_refine_enabled())

    def test_hw_config_master_switch_has_priority(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_FUSE_KERNELS": "1",
                "RTP_LLM_GLM5_PREFILL_REFINE": "1",
            },
            clear=False,
        ):
            disabled = SimpleNamespace(enable_fuse_kernels=False)
            enabled = SimpleNamespace(enable_fuse_kernels=True)
            self.assertFalse(glm5_prefill_refine_enabled(disabled))
            self.assertTrue(glm5_prefill_refine_enabled(enabled))

    def test_refine_truthy_values(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "ENABLE_FUSE_KERNELS": "1",
                    "RTP_LLM_GLM5_PREFILL_REFINE": value,
                },
                clear=False,
            ):
                self.assertTrue(glm5_prefill_refine_enabled())

    def test_refine_falsey_values(self) -> None:
        for value in ("0", "false", "FALSE", "no", "off", "invalid"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "ENABLE_FUSE_KERNELS": "1",
                    "RTP_LLM_GLM5_PREFILL_REFINE": value,
                },
                clear=False,
            ):
                self.assertFalse(glm5_prefill_refine_enabled())


if __name__ == "__main__":
    main()
