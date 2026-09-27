import copy
import math
import unittest

import torch

from src.llm.modules.surprise_delta import SurpriseDeltaMemory


class SurpriseDeltaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        torch.set_num_threads(1)

    def test_online_surprise_overwrites_same_key_and_forgets_exponentially(self):
        memory = SurpriseDeltaMemory(dim=2, memory_size=2, chunk_size=3)
        with torch.no_grad():
            memory.log_decay_fast.fill_(math.log(0.2))
            memory.logit_rate_fast.fill_(50)
            memory.read_logits.copy_(torch.tensor([50.0, -50.0]))
            memory.to_out.weight.copy_(torch.eye(2))
        q = torch.tensor([[[1.0, 0.0]] * 3])
        k = q.clone()
        v = torch.tensor([[[2.0, 0.0], [5.0, 0.0], [0.0, 0.0]]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        output, fast, _ = memory._chunk(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), q, k, v, valid)
        decay = math.exp(-0.2)
        torch.testing.assert_close(output[0, :, 0], torch.tensor([0.0, 2.0, 5.0 + 2 * (decay - 1)]))
        # The third value corrects the same key to zero; decay acts first.
        torch.testing.assert_close(fast[0, 0, 0], torch.tensor((5 + 2 * (decay - 1)) * (decay - 1)))
        empty = torch.zeros_like(q)
        _, retained, _ = memory._chunk(
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, 2),
            empty,
            empty,
            torch.zeros_like(v),
            valid,
        )
        torch.testing.assert_close(retained, torch.full_like(retained, decay**3))

    def test_mask_preserves_state_and_split_forward_matches_whole(self):
        memory = SurpriseDeltaMemory(dim=4, memory_size=3, chunk_size=2)
        memory.eval()
        inputs = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, False, True]])
        whole, final = memory(inputs, token_mask=mask, return_state=True, batched_state=True)
        first, state = memory(inputs[:, :2], return_state=True, batched_state=True)
        second, resumed = memory(
            inputs[:, 2:],
            token_mask=mask[:, 2:],
            state=state,
            return_state=True,
            batched_state=True,
        )
        torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
        for name in ("fast", "slow"):
            torch.testing.assert_close(final["params"][name], resumed["params"][name])
        torch.testing.assert_close(whole[:, 2], torch.zeros_like(whole[:, 2]))

    def test_empty_state_has_zero_read_and_meta_gradient_reaches_writer(self):
        memory = SurpriseDeltaMemory(dim=4, memory_size=3, chunk_size=2)
        inputs = torch.randn(1, 4, 4)
        output, state = memory(inputs, return_state=True, batched_state=True)
        torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
        torch.testing.assert_close(memory.read_state(inputs, None), torch.zeros_like(inputs))
        loss = memory.read_state(inputs[:, :1], state).square().sum()
        loss.backward()
        assert memory.to_k.weight.grad is not None
        assert memory.to_v.weight.grad is not None
        assert memory.to_k.weight.grad.abs().sum() > 0
        assert memory.to_v.weight.grad.abs().sum() > 0

    def test_checkpoint_and_eager_meta_gradients_agree(self):
        eager = SurpriseDeltaMemory(dim=5, memory_size=4, chunk_size=2, checkpoint_chunks=False)
        checked = copy.deepcopy(eager)
        checked.checkpoint_chunks = True
        inputs = torch.randn(2, 5, 5)
        mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        first, first_state = eager(inputs, token_mask=mask, return_state=True, batched_state=True)
        second, second_state = checked(
            inputs, token_mask=mask, return_state=True, batched_state=True
        )
        torch.testing.assert_close(first, second)
        for name in ("fast", "slow"):
            torch.testing.assert_close(first_state["params"][name], second_state["params"][name])
        first_loss = first.square().sum() + first_state["params"]["fast"].square().sum()
        second_loss = second.square().sum() + second_state["params"]["fast"].square().sum()
        first_loss.backward()
        second_loss.backward()
        for (name, parameter), (other_name, other) in zip(
            eager.named_parameters(), checked.named_parameters(), strict=True
        ):
            assert name == other_name
            torch.testing.assert_close(parameter.grad, other.grad)

    def test_batched_users_have_independent_fast_states(self):
        memory = SurpriseDeltaMemory(dim=4, memory_size=3, chunk_size=2)
        memory.eval()
        inputs = torch.randn(2, 4, 4)
        together, state = memory(inputs, return_state=True, batched_state=True)
        for user in range(2):
            alone, individual = memory(
                inputs[user : user + 1], return_state=True, batched_state=True
            )
            torch.testing.assert_close(together[user : user + 1], alone)
            for name in ("fast", "slow"):
                torch.testing.assert_close(
                    state["params"][name][user : user + 1], individual["params"][name]
                )


if __name__ == "__main__":
    unittest.main()
