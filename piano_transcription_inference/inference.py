import os
from pathlib import Path

import numpy as np
import torch

from . import config
from .models import Note_pedal
from .pytorch_utils import forward, forward_stream
from .utilities import (
    RegressionPostProcessor,
    StreamRegressionPostProcessor,
    create_folder,
    write_events_to_midi,
)


class PianoTranscription(object):
    def __init__(
        self,
        model_type="Note_pedal",
        checkpoint_path=None,
        segment_samples=16000 * 10,
        device=torch.device("cuda"),
    ):
        """Class for transcribing piano solo recording.

        Args:
          model_type: str
          checkpoint_path: str
          segment_samples: int
          device: 'cuda' | 'cpu'
        """
        if not checkpoint_path:
            checkpoint_path = "{}/.piano_transcription_inference_data/note_F1=0.9677_pedal_F1=0.9186.pth".format(
                str(Path.home())
            )
        print("Checkpoint path: {}".format(checkpoint_path))

        if (
            not os.path.exists(checkpoint_path)
            or os.path.getsize(checkpoint_path) < 1.6e8
        ):
            create_folder(os.path.dirname(checkpoint_path))
            print("Total size: ~165 MB")
            zenodo_path = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
            os.system('wget -O "{}" "{}"'.format(checkpoint_path, zenodo_path))

        print("Using {} for inference.".format(device))

        self.segment_samples = segment_samples
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Build model
        if model_type == "Note_pedal":
            Model = Note_pedal
        else:
            raise ValueError(f"model_type {model_type} not recognized")
        self.model = Model(
            frames_per_second=self.frames_per_second, classes_num=self.classes_num
        )

        # Load model
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"], strict=False)

        # Parallel
        if "cuda" in str(device):
            self.model.to(device)
            print("GPU number: {}".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print("Using CPU.")

    def transcribe(self, audio, midi_path=None):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ...}

        """
        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        pad_len = (
            int(np.ceil(audio_len / self.segment_samples)) * self.segment_samples
            - audio_len
        )

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments
        segments = self.enframe(audio, self.segment_samples)
        """(N, segment_samples)"""

        # Forward
        output_dict = forward(self.model, segments, batch_size=1)
        """{'reg_onset_output': (N, segment_frames, classes_num), ...}"""

        # Deframe to original length
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:audio_len]
        """output_dict: {
          'reg_onset_output': (N, segment_frames, classes_num),
          'reg_offset_output': (N, segment_frames, classes_num),
          'frame_output': (N, segment_frames, classes_num),
          'velocity_output': (N, segment_frames, classes_num)}"""

        # Post processor
        post_processor = RegressionPostProcessor(
            self.frames_per_second,
            classes_num=self.classes_num,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshod,
            frame_threshold=self.frame_threshold,
            pedal_offset_threshold=self.pedal_offset_threshold,
        )

        # Post process output_dict to MIDI events
        (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
            output_dict
        )

        # Write MIDI events to file
        if midi_path:
            write_events_to_midi(
                start_time=0,
                note_events=est_note_events,
                pedal_events=est_pedal_events,
                midi_path=midi_path,
            )
            print("Write out to {}".format(midi_path))

        transcribed_dict = {
            "output_dict": output_dict,
            "est_note_events": est_note_events,
            "est_pedal_events": est_pedal_events,
        }

        return transcribed_dict

    def enframe(self, x, segment_samples):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0:-1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(
                    x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)]
                )
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y

    def transcribe_stream(self, audio, midi_path=None, verbose=True):
        """Transcribe an audio recording in a streaming way.

        This function is useful for transcribing long audio recordings.
        The dataflow in the non-streaming way is as follows:

        (1)     Audio (audio_samples, )
        (2) ->  Enframe to segments (N, segment_samples)
        (3) ->  Forward (N, segment_frames, classes_num)
        (4) ->  Deframe to continuous sequences (audio_frames, classes_num)
        (5) ->  Post process to MIDI events (note_events, pedal_events)

        Step (1) to (4) require a large amount of memory when the audio is long.
        This function converts (1 - 4) to a streaming way to reduce memory consumption.

        Args:
          audio_stream: (L, buffer_samples)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'est_note_events': ..., 'est_pedal_events': ...}

        """

        frame_stream = self.enframe_stream(
            self.get_audio_stream(audio), verbose=verbose
        )
        output_dict_stream = forward_stream(self.model, frame_stream, batch_size=1)
        deframe_stream = self.deframe_stream(output_dict_stream)

        # Post processor
        post_processor = StreamRegressionPostProcessor(
            self.frames_per_second,
            classes_num=self.classes_num,
            onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshod,
            frame_threshold=self.frame_threshold,
            pedal_offset_threshold=self.pedal_offset_threshold,
        )
        (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
            deframe_stream
        )

        # Write MIDI events to file
        if midi_path:
            write_events_to_midi(
                start_time=0,
                note_events=est_note_events,
                pedal_events=est_pedal_events,
                midi_path=midi_path,
            )
            print("Write out to {}".format(midi_path))

        transcribed_dict = {
            "est_note_events": est_note_events,
            "est_pedal_events": est_pedal_events,
        }

        return transcribed_dict

    def get_audio_stream(self, audio):
        """Get audio stream from audio.

        Args:
          audio: (audio_samples, )

        Returns:
          audio_stream: (L, buffer_samples)
        """
        pointer = 0
        while pointer + self.segment_samples <= audio.shape[0]:
            yield (
                audio[pointer : pointer + self.segment_samples],
                audio.shape[0] / 16000,
            )
            pointer += self.segment_samples

        if pointer < audio.shape[0]:
            yield audio[pointer:], audio.shape[0] / 16000

    def enframe_stream(self, audio_stream, verbose=False):
        """Enframe long sequence to short segments.

        Args:
          audio_stream: (L, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """

        prev_chunk = None
        small_chunk = 0
        for i, (chunk, duration) in enumerate(audio_stream):
            chunk_time = self.segment_samples / 16000
            total_chunks = int(np.ceil(duration / chunk_time))

            if len(chunk) < self.segment_samples:
                pad_len = int(
                    np.ceil(len(chunk) / self.segment_samples)
                ) * self.segment_samples - len(chunk)
                chunk = np.concatenate((chunk, np.zeros(pad_len)))
                small_chunk += 1

            assert len(chunk) <= self.segment_samples
            assert small_chunk <= 1  # only the last chunk can be small

            if prev_chunk is not None:
                l = prev_chunk[self.segment_samples // 2 :]
                r = chunk[: self.segment_samples // 2]
                yield np.concatenate((l, r))

            yield chunk
            prev_chunk = chunk

            if verbose:
                print("Segment {} / {}".format(i + 1, total_chunks))

    def deframe_stream(self, output_dict_stream):
        """Deframe overlapped predicted segments to frame stream.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (L, segment_frames / 4, classes_num)
        """

        last_output_dict = None
        first_batch = True

        for output_dict in output_dict_stream:
            batch_size = 1
            for value in output_dict.values():
                batch_size = value.shape[0]

            for b in range(batch_size):
                if first_batch:
                    yield self._deframe_stream_chunk(output_dict, b, 0.0, 0.25)
                    first_batch = False

                yield self._deframe_stream_chunk(output_dict, b, 0.25, 0.5)
                yield self._deframe_stream_chunk(output_dict, b, 0.5, 0.75)

            last_output_dict = output_dict

        if last_output_dict is not None:
            yield self._deframe_stream_chunk(last_output_dict, -1, 0.75, 1.0)

    def _deframe_stream_chunk(self, output_dict, b, l, r):
        buf = {}
        for key in output_dict.keys():
            x = output_dict[key][b]
            x = x[0:-1, :]
            segment_samples = x.shape[0]
            assert segment_samples % 4 == 0
            buf[key] = x[int(l * segment_samples) : int(r * segment_samples)]
        return buf
