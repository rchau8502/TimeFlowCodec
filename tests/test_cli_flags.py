from test_media import make_video

from timeflowcodec.cli import compress_main, decompress_main, main


def test_cli_crf_and_tune(tmp_path):
    source = tmp_path / "source1.mp4"
    make_video(source)

    out_screen = tmp_path / "screen.tfc"
    # CLI compress_main with --crf and --tune screen
    compress_main([str(source), str(out_screen), "--crf", "18", "--tune", "screen", "--threads", "2"])
    assert out_screen.exists()
    assert out_screen.stat().st_size > 0

    out_anime = tmp_path / "anime.tfc"
    # CLI compress_main with --crf and --tune anime
    compress_main([str(source), str(out_anime), "--crf", "30", "--tune", "anime"])
    assert out_anime.exists()


def test_cli_bitrate_and_rdo(tmp_path):
    source = tmp_path / "source2.mp4"
    make_video(source)

    out_rdo = tmp_path / "rdo.tfc"
    compress_main([str(source), str(out_rdo), "--bitrate", "1500k", "--rdo", "--threads", "1"])
    assert out_rdo.exists()


def test_cli_main_subcommands(tmp_path):
    source = tmp_path / "source3.mp4"
    make_video(source)

    encoded = tmp_path / "encoded.tfc"
    # Test 'main' with 'encode' subcommand
    main(["encode", str(source), str(encoded), "--crf", "23", "--threads", "2"])
    assert encoded.exists()

    # Test 'decompress_main'
    restored = tmp_path / "restored.mp4"
    decompress_main([str(encoded), str(restored), "--threads", "2"])
    assert restored.exists()
