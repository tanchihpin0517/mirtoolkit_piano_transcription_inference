from piano_transcription_inference import PianoTranscription, load_audio, sample_rate
from piano_transcription_inference.utilities import load_audio_stream


def test_inference_stream():
    audio_path = "resources/cut_liszt.mp3"

    # Load audio
    audio_stream = load_audio_stream(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(
        device="cuda", checkpoint_path=None
    )  # device: 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    transcribed_dict_stream = transcriptor.transcribe_stream(audio_stream)

    # Load audio
    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio)

    for key in transcribed_dict_stream:
        assert len(transcribed_dict_stream[key]) == len(transcribed_dict[key])
        transcribed_dict[key] = sorted(
            transcribed_dict[key], key=lambda x: list(x.values())
        )
        transcribed_dict_stream[key] = sorted(
            transcribed_dict_stream[key], key=lambda x: list(x.values())
        )
        for item_stream, item in zip(
            transcribed_dict_stream[key], transcribed_dict[key]
        ):
            assert (
                abs(round(item_stream["onset_time"] - item["onset_time"], 2)) < 0.05
            ), (
                item_stream,
                item,
            )

            if "midi_note" in item_stream:
                assert item_stream["midi_note"] == item["midi_note"], (
                    item_stream,
                    item,
                )


if __name__ == "__main__":
    test_inference_stream()
