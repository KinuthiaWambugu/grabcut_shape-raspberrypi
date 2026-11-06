"""Offline tests for the capture prompts and batch numbering (no robot needed)."""
import pytest

pytest.importorskip("pyniryo")

from src.grabcut_shape_algorithm.camera.niryo_camera import (  # noqa: E402
    ask_continue,
    next_batch_index,
)


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


@pytest.mark.parametrize("answer", ["", "  ", "y", "Y", "yes", "YES"])
def test_enter_or_yes_continues(answer):
    assert ask_continue("go?", input_fn=scripted(answer)) is True


@pytest.mark.parametrize("answer", ["q", "Q", "quit", "n", "no", " Quit "])
def test_q_or_no_quits(answer):
    assert ask_continue("go?", input_fn=scripted(answer)) is False


def test_unrecognised_input_asks_again():
    assert ask_continue("go?", input_fn=scripted("maybe", "???", "q")) is False
    assert ask_continue("go?", input_fn=scripted("maybe", "")) is True


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_closed_input_or_ctrl_c_quits(exc):
    def raiser(prompt):
        raise exc
    assert ask_continue("go?", input_fn=raiser) is False


def test_prompt_text_mentions_both_options():
    seen = []
    ask_continue("Capture?", input_fn=lambda p: seen.append(p) or "q")
    assert "Enter" in seen[0] and "q" in seen[0]


def test_next_batch_index_empty_or_missing_folder(tmp_path):
    assert next_batch_index(tmp_path, "scene") == 1
    assert next_batch_index(tmp_path / "does_not_exist", "scene") == 1


def test_next_batch_index_continues_after_existing_batches(tmp_path):
    for name in ["scene01_P1.png", "scene01_HOME.png", "scene03_P1.png", "background07_P1.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert next_batch_index(tmp_path, "scene") == 4          # highest scene batch is 03
    assert next_batch_index(tmp_path, "background") == 8     # kinds are numbered separately
    assert next_batch_index(tmp_path, "other") == 1