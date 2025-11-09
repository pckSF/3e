"""Test script for ReplayBuffer, especially add_transitions method."""

from __future__ import annotations

import numpy as np

from scs.buffer import ReplayBuffer


def test_add_single_transition():
    """Test adding single transitions."""
    buffer = ReplayBuffer(max_size=10, state_dim=4, action_dim=2)

    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5, -0.5])
    reward = 1.5
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    terminal = False

    buffer.add_transition(state, action, reward, next_state, terminal)

    assert buffer.current_index == 1
    assert buffer.available_samples == 1
    assert not buffer.full
    assert np.allclose(buffer.states[0], state)
    assert np.allclose(buffer.actions[0], action)
    assert buffer.rewards[0] == reward
    assert np.allclose(buffer.next_states[0], next_state)
    assert buffer.terminals[0] == terminal

    print("✓ test_add_single_transition passed")


def test_add_transitions_no_wrap():
    """Test adding batch of transitions without wrapping."""
    buffer = ReplayBuffer(max_size=100, state_dim=4, action_dim=2)

    n = 10
    states = np.random.randn(n, 4).astype(np.float32)
    actions = np.random.randn(n, 2).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    next_states = np.random.randn(n, 4).astype(np.float32)
    terminals = np.random.rand(n) > 0.5

    buffer.add_transitions(states, actions, rewards, next_states, terminals)

    assert buffer.current_index == 10
    assert buffer.available_samples == 10
    assert not buffer.full
    assert np.allclose(buffer.states[:10], states)
    assert np.allclose(buffer.actions[:10], actions)
    assert np.allclose(buffer.rewards[:10], rewards)
    assert np.allclose(buffer.next_states[:10], next_states)
    assert np.allclose(buffer.terminals[:10], terminals)

    print("✓ test_add_transitions_no_wrap passed")


def test_add_transitions_with_wrap():
    """Test adding batch that wraps around the buffer."""
    buffer = ReplayBuffer(max_size=10, state_dim=4, action_dim=2)

    # Fill buffer to position 7
    for i in range(7):
        buffer.add_transition(
            np.array([i, i, i, i], dtype=np.float32),
            np.array([i, i], dtype=np.float32),
            float(i),
            np.array([i + 1, i + 1, i + 1, i + 1], dtype=np.float32),
            False,
        )

    assert buffer.current_index == 7
    assert buffer.available_samples == 7

    # Add 5 more transitions (will wrap: 3 at end, 2 at start)
    n = 5
    states = np.arange(n * 4).reshape(n, 4).astype(np.float32)
    actions = np.arange(n * 2).reshape(n, 2).astype(np.float32)
    rewards = np.arange(n).astype(np.float32)
    next_states = np.arange(100, 100 + n * 4).reshape(n, 4).astype(np.float32)
    terminals = np.array([False, True, False, True, False])

    buffer.add_transitions(states, actions, rewards, next_states, terminals)

    # Should wrap: indices 7, 8, 9 get first 3, then indices 0, 1 get last 2
    assert buffer.current_index == 2, f"Expected index 2, got {buffer.current_index}"
    assert buffer.full
    assert buffer.available_samples == 10

    # Check first chunk (indices 7, 8, 9)
    assert np.allclose(buffer.states[7:10], states[:3])
    assert np.allclose(buffer.actions[7:10], actions[:3])
    assert np.allclose(buffer.rewards[7:10], rewards[:3])
    assert np.allclose(buffer.next_states[7:10], next_states[:3])
    assert np.allclose(buffer.terminals[7:10], terminals[:3])

    # Check wrapped chunk (indices 0, 1)
    assert np.allclose(buffer.states[0:2], states[3:5])
    assert np.allclose(buffer.actions[0:2], actions[3:5])
    assert np.allclose(buffer.rewards[0:2], rewards[3:5])
    assert np.allclose(buffer.next_states[0:2], next_states[3:5])
    assert np.allclose(buffer.terminals[0:2], terminals[3:5])

    print("✓ test_add_transitions_with_wrap passed")


def test_add_transitions_exact_fit():
    """Test adding transitions that exactly fill to max_size."""
    buffer = ReplayBuffer(max_size=10, state_dim=4, action_dim=2)

    # Add exactly 10 transitions
    n = 10
    states = np.random.randn(n, 4).astype(np.float32)
    actions = np.random.randn(n, 2).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    next_states = np.random.randn(n, 4).astype(np.float32)
    terminals = np.random.rand(n) > 0.5

    buffer.add_transitions(states, actions, rewards, next_states, terminals)

    assert buffer.current_index == 0, f"Expected index 0, got {buffer.current_index}"
    assert buffer.full
    assert buffer.available_samples == 10

    print("✓ test_add_transitions_exact_fit passed")


def test_add_transitions_partial_then_wrap():
    """Test adding transitions in two batches that wrap."""
    buffer = ReplayBuffer(max_size=10, state_dim=4, action_dim=2)

    # Add 6 transitions
    n1 = 6
    states1 = np.ones((n1, 4), dtype=np.float32)
    actions1 = np.ones((n1, 2), dtype=np.float32)
    rewards1 = np.ones(n1, dtype=np.float32)
    next_states1 = np.ones((n1, 4), dtype=np.float32) * 2
    terminals1 = np.zeros(n1, dtype=bool)

    buffer.add_transitions(states1, actions1, rewards1, next_states1, terminals1)

    assert buffer.current_index == 6
    assert buffer.available_samples == 6
    assert not buffer.full

    # Add 6 more (will wrap: 4 at end, 2 at start)
    n2 = 6
    states2 = np.ones((n2, 4), dtype=np.float32) * 3
    actions2 = np.ones((n2, 2), dtype=np.float32) * 3
    rewards2 = np.ones(n2, dtype=np.float32) * 3
    next_states2 = np.ones((n2, 4), dtype=np.float32) * 4
    terminals2 = np.ones(n2, dtype=bool)

    buffer.add_transitions(states2, actions2, rewards2, next_states2, terminals2)

    assert buffer.current_index == 2, f"Expected index 2, got {buffer.current_index}"
    assert buffer.full
    assert buffer.available_samples == 10

    # Verify the data
    # Indices 0-1: from second batch (last 2 elements)
    assert np.allclose(buffer.states[0:2], states2[4:6])
    # Indices 2-5: from first batch (elements 2-5, since 0-1 were overwritten)
    assert np.allclose(buffer.states[2:6], states1[2:6])
    # Indices 6-9: from second batch (first 4 elements)
    assert np.allclose(buffer.states[6:10], states2[0:4])

    print("✓ test_add_transitions_partial_then_wrap passed")


def test_buffer_wrapping_multiple_times():
    """Test that buffer correctly handles multiple wraps."""
    buffer = ReplayBuffer(max_size=5, state_dim=2, action_dim=1)

    # Fill the buffer
    for i in range(5):
        buffer.add_transition(
            np.array([i, i], dtype=np.float32),
            np.array([i], dtype=np.float32),
            float(i),
            np.array([i + 1, i + 1], dtype=np.float32),
            False,
        )

    assert buffer.full
    assert buffer.current_index == 0

    # Add more transitions one by one
    for i in range(5, 10):
        buffer.add_transition(
            np.array([i, i], dtype=np.float32),
            np.array([i], dtype=np.float32),
            float(i),
            np.array([i + 1, i + 1], dtype=np.float32),
            i % 2 == 0,
        )

    # Should be back at index 0
    assert buffer.current_index == 0
    assert buffer.full
    assert buffer.available_samples == 5

    # The buffer should contain the last 5 transitions (5-9)
    for i in range(5):
        expected_val = i + 5
        assert np.allclose(buffer.states[i], [expected_val, expected_val])
        assert np.allclose(buffer.actions[i], [expected_val])
        assert buffer.rewards[i] == expected_val

    print("✓ test_buffer_wrapping_multiple_times passed")


def test_error_on_too_large_batch():
    """Test that adding too many transitions raises an error."""
    buffer = ReplayBuffer(max_size=10, state_dim=4, action_dim=2)

    n = 15  # More than buffer size
    states = np.random.randn(n, 4).astype(np.float32)
    actions = np.random.randn(n, 2).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    next_states = np.random.randn(n, 4).astype(np.float32)
    terminals = np.random.rand(n) > 0.5

    try:
        buffer.add_transitions(states, actions, rewards, next_states, terminals)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds buffer maximum size" in str(e)

    print("✓ test_error_on_too_large_batch passed")


def test_edge_case_wrap_at_exactly_max():
    """Test edge case where batch ends exactly at max_size."""
    buffer = ReplayBuffer(max_size=10, state_dim=2, action_dim=1)

    # Add 7 transitions first
    for i in range(7):
        buffer.add_transition(
            np.array([i, i], dtype=np.float32),
            np.array([i], dtype=np.float32),
            float(i),
            np.array([i + 1, i + 1], dtype=np.float32),
            False,
        )

    # Add exactly 3 more to reach max_size
    n = 3
    states = np.arange(n * 2).reshape(n, 2).astype(np.float32)
    actions = np.arange(n).reshape(n, 1).astype(np.float32)
    rewards = np.arange(n).astype(np.float32)
    next_states = np.arange(100, 100 + n * 2).reshape(n, 2).astype(np.float32)
    terminals = np.array([False, True, False])

    buffer.add_transitions(states, actions, rewards, next_states, terminals)

    # Should wrap to 0 and mark as full
    assert buffer.current_index == 0, f"Expected index 0, got {buffer.current_index}"
    assert buffer.full
    assert buffer.available_samples == 10

    # Verify data at indices 7, 8, 9
    assert np.allclose(buffer.states[7:10], states)

    print("✓ test_edge_case_wrap_at_exactly_max passed")


if __name__ == "__main__":
    print("Running ReplayBuffer tests...\n")

    test_add_single_transition()
    test_add_transitions_no_wrap()
    test_add_transitions_with_wrap()
    test_add_transitions_exact_fit()
    test_add_transitions_partial_then_wrap()
    test_buffer_wrapping_multiple_times()
    test_error_on_too_large_batch()
    test_edge_case_wrap_at_exactly_max()

    print("\n✅ All tests passed!")
