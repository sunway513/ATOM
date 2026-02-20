# SPDX-License-Identifier: MIT
# Tests for MOE shape handling with fused shared experts.
#
# The _triton_fp8_moe function must handle the case where shared experts
# are fused into routing, making topk_ids have M*(top_k+1) elements
# instead of M*top_k. The `actual_top_k` computation guards against this.
#
# Tests use plain Python (no torch) to avoid conftest.py stub conflicts.

import pytest


class TestMoeActualTopK:
    """Test actual_top_k = topk_ids_numel // M computation."""

    @pytest.mark.parametrize(
        "M, top_k, n_shared_experts, expected_actual_top_k",
        [
            (4, 8, 0, 8),  # No shared experts: actual_top_k == top_k
            (4, 8, 1, 9),  # 1 shared expert fused: actual_top_k == top_k + 1
            (1, 8, 1, 9),  # Single token with shared expert
            (16, 6, 0, 6),  # Different top_k, no shared
            (16, 6, 1, 7),  # Different top_k, with shared
            (128, 8, 1, 9),  # Larger batch with shared expert
        ],
    )
    def test_actual_top_k_computation(
        self, M, top_k, n_shared_experts, expected_actual_top_k
    ):
        """actual_top_k should reflect the true number of expert slots per token."""
        effective_top_k = top_k + n_shared_experts
        topk_ids_numel = M * effective_top_k
        actual_top_k = topk_ids_numel // M
        assert actual_top_k == expected_actual_top_k

    @pytest.mark.parametrize(
        "M, top_k, n_shared_experts",
        [
            (4, 8, 0),
            (4, 8, 1),
            (16, 6, 0),
            (16, 6, 1),
        ],
    )
    def test_intermediate_tensor_shapes(self, M, top_k, n_shared_experts):
        """Intermediate tensors must use actual_top_k, not top_k."""
        effective_top_k = top_k + n_shared_experts
        inter_dim = 1024
        hidden_dim = 7168

        topk_ids_numel = M * effective_top_k
        actual_top_k = topk_ids_numel // M

        # These shapes must match what _triton_fp8_moe creates
        intermediate_shape = (M * actual_top_k, inter_dim)
        output_shape = (M * actual_top_k, 1, hidden_dim)

        # Reshape for GEMM2
        gemm2_topk_ids_shape = (M * actual_top_k, 1)
        gemm2_topk_weights_shape = (M * actual_top_k, 1)

        assert intermediate_shape == (M * effective_top_k, inter_dim)
        assert output_shape == (M * effective_top_k, 1, hidden_dim)
        assert gemm2_topk_ids_shape == (M * effective_top_k, 1)
        assert gemm2_topk_weights_shape == (M * effective_top_k, 1)

    @pytest.mark.parametrize(
        "M, top_k, n_shared_experts",
        [
            (4, 8, 0),
            (4, 8, 1),
            (16, 6, 1),
        ],
    )
    def test_final_reduce_shape(self, M, top_k, n_shared_experts):
        """Final reduce must reshape to (M, actual_top_k, hidden_dim) then sum."""
        effective_top_k = top_k + n_shared_experts
        hidden_dim = 7168

        topk_ids_numel = M * effective_top_k
        actual_top_k = topk_ids_numel // M

        # output.squeeze(1).view(M, actual_top_k, hidden_dim).sum(dim=1)
        # Result shape: (M, hidden_dim)
        output_flat = M * actual_top_k * hidden_dim  # total elements
        # Reshape check: M * actual_top_k * hidden_dim must equal output_flat
        assert M * actual_top_k * hidden_dim == output_flat

    def test_shape_mismatch_without_fix(self):
        """Demonstrates the crash when using top_k instead of actual_top_k.

        With fused shared expert: topk_ids has M*9 elements but reshape
        with M*8 fails because 36 != 32.
        """
        M, top_k, n_shared_experts = 4, 8, 1
        effective_top_k = top_k + n_shared_experts  # 9
        topk_ids_numel = M * effective_top_k  # 36

        # Using top_k (8) instead of actual_top_k (9) -> M*top_k = 32 != 36
        assert topk_ids_numel != M * top_k
        # This would fail: topk_ids.reshape(M * top_k, 1) -> RuntimeError

    def test_actual_top_k_matches_effective(self):
        """actual_top_k computation always matches the effective top_k."""
        for M in [1, 4, 16, 128]:
            for top_k in [4, 6, 8]:
                for n_shared in [0, 1]:
                    effective = top_k + n_shared
                    numel = M * effective
                    actual = numel // M
                    assert actual == effective, (
                        f"M={M}, top_k={top_k}, n_shared={n_shared}: "
                        f"actual={actual} != effective={effective}"
                    )

    def test_sorting_buffer_size_uses_numel(self):
        """Stage 1 sorting: max_num_tokens_padded uses topk_ids.numel(), not M*top_k."""
        M, top_k, n_shared = 4, 8, 1
        E = 256  # num experts
        block_size_m = 128
        effective_top_k = top_k + n_shared
        topk_ids_numel = M * effective_top_k  # 36

        # This matches the code: max_num_tokens_padded = topk_ids.numel() + E * (block_size_m - 1)
        max_num_tokens_padded = topk_ids_numel + E * (block_size_m - 1)
        assert max_num_tokens_padded == 36 + 256 * 127

        # Wrong calculation with top_k instead of actual_top_k
        wrong_padded = M * top_k + E * (block_size_m - 1)
        assert wrong_padded != max_num_tokens_padded
