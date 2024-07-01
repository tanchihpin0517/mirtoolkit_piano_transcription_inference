from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from piano_transcription_inference.utilities import load_audio_stream


def test_inference_stream():
    audio_path='resources/cut_liszt.mp3'

    # Load audio
    audio_stream = load_audio_stream(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)  # device: 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe_stream(audio_stream)


if __name__ == '__main__':
    test_inference_stream()
