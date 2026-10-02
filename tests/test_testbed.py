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
