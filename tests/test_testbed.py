import tempfile
from pathlib import Path

from testbed.api.webhook import EDU_EMAIL_PATTERN, hash_telegram_id
from testbed.hpc.hpc_worker import run_research_codec


def test_zero_pii_telegram_id_hashing():
    """Verify Telegram ID hashing is deterministic and non-reversible."""
    raw_id = 987654321
    hashed1 = hash_telegram_id(raw_id)
    hashed2 = hash_telegram_id(raw_id)
    diff_hash = hash_telegram_id(987654322)

    assert len(hashed1) == 64
    assert hashed1 == hashed2
    assert hashed1 != diff_hash
    # Ensure raw ID is never present in hash
    assert str(raw_id) not in hashed1


def test_edu_email_validation():
    """Verify email domain regex strictly accepts university affiliations and rejects non-edu."""
    valid_emails = [
        "student@stanford.edu",
        "researcher@cs.cmu.edu",
        "phd@cam.ac.uk",
        "fellow@eng.ox.ac.uk",
        "scholar@anu.edu.au",
    ]
    invalid_emails = [
        "user@gmail.com",
        "user@yahoo.com",
        "attacker@eduscams.com",
        "hacker@fake-edu.org",
        "not_an_email",
    ]

    for email in valid_emails:
        assert EDU_EMAIL_PATTERN.match(email), f"Failed to match valid university email: {email}"

    for email in invalid_emails:
        assert not EDU_EMAIL_PATTERN.match(email), f"Incorrectly accepted invalid email: {email}"


def test_hpc_research_codec_execution():
    """Verify HPC worker codec execution produces valid output and telemetry."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_file = tmp_path / "sample_video.mp4"
        output_file = tmp_path / "result.tfc"

        # Create dummy input payload
        input_file.write_bytes(b"TEST_VIDEO_STREAM_DATA_" * 100)

        metrics = run_research_codec(input_file, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0
        assert "compression_ratio" in metrics
        assert "execution_time_sec" in metrics
        assert metrics["input_bytes"] == input_file.stat().st_size
        assert metrics["output_bytes"] == output_file.stat().st_size


def test_hpc_worker_claim_and_completion_pipeline(monkeypatch):
    """Verify full HPC polling, claim, execute, and completion reporting pipeline."""
    from unittest.mock import MagicMock

    import testbed.hpc.hpc_worker as worker

    mock_supabase = MagicMock()
    mock_task = {
        "id": "task-uuid-1234",
        "media_storage_path": "uploads/user_hash/clip.mp4",
        "user_hash": "user_hash_abc",
        "target_engine": "TimeFlowCodec_v2",
    }

    mock_rpc = MagicMock()
    mock_rpc.execute.return_value.data = [mock_task]
    mock_supabase.rpc.return_value = mock_rpc

    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value.data = [mock_task]
    mock_supabase.table.return_value = mock_table

    monkeypatch.setattr(worker, "supabase", mock_supabase)

    claimed = worker.claim_next_pending_task()
    assert claimed == mock_task
    mock_supabase.rpc.assert_called_with("claim_next_hpc_task", {"p_worker_id": worker.WORKER_ID})

    metrics = {"compression_ratio": 2.5, "execution_time_sec": 0.4}
    worker.mark_task_completed("task-uuid-1234", "https://storage/result.tfc", metrics)
    mock_supabase.table.assert_called_with("hpc_tasks")


def test_hpc_worker_signal_shutdown():
    """Verify SIGTERM/SIGINT signal triggers graceful shutdown flag."""
    import signal

    import testbed.hpc.hpc_worker as worker

    worker.RUNNING = True
    worker.handle_shutdown(signal.SIGTERM, None)
    assert worker.RUNNING is False
    worker.RUNNING = True  # reset for worker lifecycle

