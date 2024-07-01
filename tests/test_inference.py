from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from piano_transcription_inference.utilities import load_audio_stream


def test_inference():
    audio_path='resources/cut_liszt.mp3'

    # Load audio
    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)  # device: 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio)


if __name__ == '__main__':
    test_inference()
