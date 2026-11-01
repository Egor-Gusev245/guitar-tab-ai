"""
Pipeline CLI: End-to-end Guitar Tablature Generator
Separates guitar stem with Demucs and generates guitar tabs using Spotify Basic Pitch.
"""

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
from tab_extractor import GuitarTabExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract guitar tabs from a song snippet using Demucs and Spotify Basic Pitch."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Path to the input audio file (MP3, WAV, FLAC, etc.).",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=float,
        default=0.0,
        help="Start time of the snippet in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=float,
        default=None,
        help="End time of the snippet in seconds.",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Duration of the snippet in seconds (used if --end is not specified).",
    )
    parser.add_argument(
        "--is-guitar-stem",
        action="store_true",
        help="Set this flag if the input file is already an isolated guitar audio (skips Demucs).",
    )
    parser.add_argument(
        "--full-demux",
        action="store_true",
        help="Run Demucs on the entire track and cache the full guitar stem.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Demucs ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--save-tab",
        type=str,
        default=None,
        help="Path to save the generated ASCII tab file (e.g. output/tab.txt).",
    )
    parser.add_argument(
        "--save-midi",
        type=str,
        default=None,
        help="Path to save the transcribed MIDI file (e.g. output/solo.mid).",
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        help="Path to save the isolated guitar snippet audio (e.g. output/snippet.wav).",
    )
    parser.add_argument(
        "--chords",
        action="store_true",
        help="Analyze harmony and display the chord progression timeline.",
    )
    parser.add_argument(
        "--save-chords",
        type=str,
        default=None,
        help="Path to save the detected chord progression JSON (e.g. output/chords.json).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum note confidence threshold for Basic Pitch (default: 0.25).",
    )
    parser.add_argument(
        "--max-span",
        type=int,
        default=4,
        help="Maximum fret stretch span for simultaneous notes (default: 4 frets).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=70,
        help="Maximum width of ASCII tab lines (default: 70).",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Determine snippet start and end time
    with sf.SoundFile(str(audio_path)) as f:
        total_duration = len(f) / f.samplerate

    start_time = max(0.0, args.start)

    if args.end is not None:
        end_time = min(total_duration, args.end)
    elif args.duration is not None:
        end_time = min(total_duration, start_time + args.duration)
    else:
        # Default snippet: 10 seconds or full track if shorter than 10s
        end_time = min(total_duration, start_time + 10.0)

    if end_time <= start_time:
        print(
            f"Error: Invalid time range [{start_time:.2f}s - {end_time:.2f}s]. "
            f"Total track duration is {total_duration:.2f}s.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 65)
    print(f"Guitar Tablature Extractor (Demucs + Spotify Basic Pitch)")
    print(f"Audio file    : {audio_path.name}")
    print(f"Total length  : {total_duration:.2f} s")
    print(f"Target snippet: [{start_time:.2f} s -> {end_time:.2f} s] ({end_time - start_time:.2f} s)")
    print("=" * 65)

    start_proc = time.time()

    # Step 1: Guitar extraction
    if args.is_guitar_stem:
        print("\n[Input] Audio flagged as direct guitar stem. Slicing snippet directly...")
        snippet_guitar_path = audio_path.parent / f"{audio_path.stem}_slice_{start_time:.2f}_{end_time:.2f}.wav"
        from stem_extractor import slice_audio
        slice_audio(audio_path, snippet_guitar_path, start_time, end_time)
    else:
        from stem_extractor import GuitarStemExtractor
        stem_extractor = GuitarStemExtractor(device=args.device)
        snippet_guitar_path = stem_extractor.get_guitar_snippet(
            audio_path,
            start_time=start_time,
            end_time=end_time,
            full_track_demux=args.full_demux,
        )

    # Step 2: Basic Pitch Transcription & Tab Generation
    tab_extractor = GuitarTabExtractor(
        min_confidence=args.confidence,
        max_hand_span=args.max_span,
        max_line_width=args.line_width,
    )
    result = tab_extractor.transcribe(snippet_guitar_path)

    elapsed = time.time() - start_proc
    print(f"\nProcessing completed in {elapsed:.2f} seconds.")
    print(f"Extracted {len(result.notes)} notes.")

    # Step 3: Print and save results
    print("\n" + "=" * 30 + " GUITAR TAB " + "=" * 30)
    print(result.ascii_tab)
    print("=" * 72)

    # Step 4: Chord Harmony Detection (if requested)
    if args.chords or args.save_chords:
        print("\n" + "=" * 27 + " CHORD HARMONY " + "=" * 27)
        from chord_detector import ChordDetector
        detector = ChordDetector()
        t_chord0 = time.time()
        chord_timeline = detector.detect_from_file(audio_path)
        print(f"Chord recognition finished in {time.time()-t_chord0:.2f}s:")
        chord_timeline.print_summary()

        if args.save_chords:
            chord_timeline.save_json(args.save_chords)

    if args.save_tab:
        result.save_tab(Path(args.save_tab))

    if args.save_midi:
        result.save_midi(Path(args.save_midi))

    if args.save_audio:
        out_audio = Path(args.save_audio)
        out_audio.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(str(snippet_guitar_path), str(out_audio))
        print(f"[Audio] Saved guitar snippet to {out_audio}")


if __name__ == "__main__":
    main()
