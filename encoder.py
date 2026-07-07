# =============================================================
# encoding.py  —  Modular audio → spike encoders
# =============================================================
# To add a new encoder:
#   1. Subclass BaseEncoder
#   2. Implement encode(audio) → np.ndarray of shape (n_mels, T_frames)
#      (or whatever shape your architecture expects)
#   3. Pass an instance to PCGSpikeDataset(encoder=YourEncoder(...))
# =============================================================
import numpy as np
import torch
import librosa
from abc import ABC, abstractmethod


# ── Shared rate-encoding helper ────────────────────────────────
def rate_encode(spec: np.ndarray, T_sim: int = 50) -> torch.Tensor:
    """
    Convert a 2-D spectrogram (n_channels, T_frames) into a spike tensor.

    Rate encoding: treat each normalised amplitude as a Bernoulli
    probability and sample T_sim binary spike trains.

    Returns: FloatTensor  (T_sim, n_channels, T_frames)
    """
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min < 1e-8:
        spec_norm = np.zeros_like(spec)
    else:
        spec_norm = (spec - s_min) / (s_max - s_min)

    prob   = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
    prob   = prob.expand(T_sim, -1, -1)          # (T_sim, C, T)
    spikes = torch.bernoulli(prob)
    return spikes                                 # (T_sim, C, T)


# ── Abstract base class ────────────────────────────────────────
class BaseEncoder(ABC):
    """
    Every encoder must implement:
        encode(audio: np.ndarray) -> np.ndarray   shape (n_channels, T_frames)

    The dataset will call rate_encode() on the returned array automatically.
    If you want a completely custom spike-generation strategy, override
    encode_to_spikes() instead and ignore rate_encode().
    """

    @abstractmethod
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """Return a 2-D feature map (n_channels, T_frames)."""
        ...

    def encode_to_spikes(self, audio: np.ndarray, T_sim: int) -> torch.Tensor:
        """
        Full pipeline: audio → feature map → spike tensor.
        Override this only if you need a non-rate encoding strategy.
        """
        feature_map = self.encode(audio)
        return rate_encode(feature_map, T_sim=T_sim)

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of output channels (used to configure the model)."""
        ...


# ── Mel spectrogram encoder (current default) ─────────────────
class MelEncoder(BaseEncoder):
    """
    Log-Mel spectrogram → rate-encoded spikes.

    Args:
        sr       : sampling rate
        n_mels   : number of Mel filterbanks  (= SNN input channels)
        n_fft    : FFT window size
        hop_len  : hop length in samples
        fmin     : lowest frequency (Hz)
        fmax     : highest frequency (Hz)
    """
    def __init__(self, sr: int = 2000, n_mels: int = 24,
                 n_fft: int = 512, hop_len: int = 128,
                 fmin: float = 20.0, fmax: float = 500.0):
        self.sr      = sr
        self.n_mels  = n_mels
        self.n_fft   = n_fft
        self.hop_len = hop_len
        self.fmin    = fmin
        self.fmax    = fmax

    def encode(self, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_len,
            n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        return librosa.power_to_db(mel, ref=np.max)   # (n_mels, T_frames)

    @property
    def n_channels(self) -> int:
        return self.n_mels

    def __repr__(self):
        return (f"MelEncoder(sr={self.sr}, n_mels={self.n_mels}, "
                f"n_fft={self.n_fft}, hop_len={self.hop_len}, "
                f"fmin={self.fmin}, fmax={self.fmax})")


# ── MFCC encoder (ready to use) ───────────────────────────────
class MFCCEncoder(BaseEncoder):
    """
    MFCC features → rate-encoded spikes.

    Args:
        sr      : sampling rate
        n_mfcc  : number of MFCC coefficients  (= SNN input channels)
        n_fft   : FFT window size
        hop_len : hop length in samples
    """
    def __init__(self, sr: int = 2000, n_mfcc: int = 24,
                 n_fft: int = 512, hop_len: int = 128):
        self.sr      = sr
        self.n_mfcc  = n_mfcc
        self.n_fft   = n_fft
        self.hop_len = hop_len

    def encode(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=audio, sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_len
        )                                              # (n_mfcc, T_frames)

    @property
    def n_channels(self) -> int:
        return self.n_mfcc

    def __repr__(self):
        return (f"MFCCEncoder(sr={self.sr}, n_mfcc={self.n_mfcc}, "
                f"n_fft={self.n_fft}, hop_len={self.hop_len})")


# ── CQT encoder (ready to use) ────────────────────────────────
class CQTEncoder(BaseEncoder):
    """
    Constant-Q Transform magnitude → rate-encoded spikes.
    CQT has logarithmic frequency resolution — good for tonal PCG sounds.

    Args:
        sr       : sampling rate
        n_bins   : total CQT frequency bins  (= SNN input channels)
        hop_len  : hop length in samples
        fmin     : lowest frequency (Hz), default C2 ≈ 32.7 Hz
        bins_per_octave : frequency resolution
    """
    def __init__(self, sr: int = 2000, n_bins: int = 24,
                 hop_len: int = 128, fmin: float = 32.7,
                 bins_per_octave: int = 12):
        self.sr              = sr
        self.n_bins          = n_bins
        self.hop_len         = hop_len
        self.fmin            = fmin
        self.bins_per_octave = bins_per_octave

    def encode(self, audio: np.ndarray) -> np.ndarray:
        cqt = librosa.cqt(
            y=audio, sr=self.sr,
            hop_length=self.hop_len,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)  # (n_bins, T)

    @property
    def n_channels(self) -> int:
        return self.n_bins

    def __repr__(self):
        return (f"CQTEncoder(sr={self.sr}, n_bins={self.n_bins}, "
                f"hop_len={self.hop_len}, fmin={self.fmin})")


# ── CWT encoder ───────────────────────────────────────────────
class CWTEncoder(BaseEncoder):
    """
    Continuous Wavelet Transform scalogram → rate-encoded spikes.

    CWT gives fine time resolution at high frequencies and fine frequency
    resolution at low frequencies — well-suited for transient PCG events
    like S1/S2 clicks that are poorly captured by fixed-window transforms.

    Uses PyWavelets (pywt). Install with:  pip install PyWavelets

    How the output shape is controlled
    ───────────────────────────────────
    CWT computes one row per scale, at every sample → shape (n_scales, N_samples).
    Since N_samples can be tens of thousands, the time axis is decimated by
    `decimation` (default 128, same as hop_len in the mel encoder) to give a
    manageable (n_scales, T_frames) output that matches what the Conv1D SNN
    expects.

    Args:
        sr          : sampling rate
        n_scales    : number of wavelet scales  (= SNN input channels)
        wavelet     : PyWavelets wavelet name.
                      'morl'  (Morlet)   — good general-purpose choice for audio
                      'mexh'  (Mexican hat / Ricker) — good for sharp transients
                      'cmor1.5-1.0'     — complex Morlet, wider bandwidth
        fmin        : lowest centre frequency (Hz) to analyse
        fmax        : highest centre frequency (Hz) to analyse
        decimation  : time-axis downsampling factor (like hop_len in mel)
    """
    def __init__(self, sr: int = 2000, n_scales: int = 24,
                 wavelet: str = 'morl',
                 fmin: float = 20.0, fmax: float = 500.0,
                 decimation: int = 128):
        self.sr         = sr
        self.n_scales   = n_scales
        self.wavelet    = wavelet
        self.fmin       = fmin
        self.fmax       = fmax
        self.decimation = decimation

        # Pre-compute scales from the requested frequency range.
        # pywt.scale2frequency(wavelet, scale) = f  →  scale = f_c / (f * dt)
        # where f_c is the wavelet centre frequency and dt = 1/sr.
        try:
            import pywt
        except ImportError:
            raise ImportError(
                "PyWavelets is required for CWTEncoder.\n"
                "Install it with:  pip install PyWavelets"
            )
        fc = pywt.central_frequency(wavelet)
        dt = 1.0 / sr
        # scale = fc / (frequency * dt)
        scale_min = fc / (fmax * dt)   # small scale → high frequency
        scale_max = fc / (fmin * dt)   # large scale → low frequency
        self._scales = np.linspace(scale_min, scale_max, n_scales)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        import pywt

        # CWT → complex or real coefficients: shape (n_scales, N_samples)
        coeffs, _ = pywt.cwt(audio, self._scales, self.wavelet,
                              sampling_period=1.0 / self.sr)

        # Take magnitude (handles both real and complex wavelets)
        scalogram = np.abs(coeffs)                    # (n_scales, N_samples)

        # Decimate time axis so T_frames ≈ N_samples // decimation
        scalogram = scalogram[:, ::self.decimation]   # (n_scales, T_frames)

        # Convert to dB — same log-scale treatment as mel/CQT
        scalogram = 20.0 * np.log10(scalogram + 1e-9)

        return scalogram                              # (n_scales, T_frames)

    @property
    def n_channels(self) -> int:
        return self.n_scales

    def __repr__(self):
        return (f"CWTEncoder(sr={self.sr}, n_scales={self.n_scales}, "
                f"wavelet='{self.wavelet}', fmin={self.fmin}, "
                f"fmax={self.fmax}, decimation={self.decimation})")


# ── Chirplet Transform encoder ────────────────────────────────
class ChirpletEncoder(BaseEncoder):
    """
    Synchrosqueezed Chirplet Transform scalogram → rate-encoded spikes.

    Why chirplets for PCG?
    ──────────────────────
    Heart sounds contain frequency-modulated (FM) components — S1/S2
    clicks and murmurs whose instantaneous frequency sweeps over time.
    A chirplet atom has a centre frequency AND a chirp rate (df/dt),
    so it aligns with FM ridges that sinusoidal transforms smear.

    Implementation
    ──────────────
    Uses PyWavelets (pywt) — already required by CWTEncoder, no new deps.
    ssqueezepy is NOT used; it has a Python 3.13 incompatibility in its
    internal scale-inference code (TypeError in process_scales) that
    cannot be worked around from outside the library.

    Pipeline per audio segment:
      1. Complex Morlet CWT via pywt           → W(scale, t)
      2. Instantaneous frequency estimation    → omega(scale, t)
             omega = Im[dW/dt / W] / 2π
         approximated by central finite difference along time axis.
      3. Energy reassignment (synchrosqueezing):
         accumulate |W(s,t)| into the output bin whose centre frequency
         matches omega(s,t). This sharpens diffuse CWT ridges into tight
         FM tracks — the chirplet effect.
      4. Decimate time axis by `decimation`, convert to dB.

    Requires:  pip install PyWavelets

    Args:
        sr          : sampling rate
        n_scales    : number of output frequency bins (= SNN input channels)
        fmin        : lowest frequency to analyse (Hz)
        fmax        : highest frequency to analyse (Hz)
        decimation  : time-axis downsampling factor (matches hop_len in mel)
        n_internal  : internal CWT scales before reassignment (>= n_scales)
    """
    def __init__(self, sr: int = 2000, n_scales: int = 24,
                 fmin: float = 20.0, fmax: float = 500.0,
                 decimation: int = 128,
                 n_internal: int = 96):
        self.sr         = sr
        self.n_scales   = n_scales
        self.fmin       = fmin
        self.fmax       = fmax
        self.decimation = decimation
        self.n_internal = max(n_internal, n_scales * 4)

        try:
            import pywt  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyWavelets is required for ChirpletEncoder.\n"
                "Install with:  pip install PyWavelets"
            )

        # Pre-compute internal CWT scales covering [fmin, fmax]
        # pywt scale-to-frequency:  f = fc / (scale * dt)  →  scale = fc / (f * dt)
        import pywt as _pywt
        self._wavelet = 'cmor1.5-1.0'        # complex Morlet — needed for IF estimate
        fc            = _pywt.central_frequency(self._wavelet)
        dt            = 1.0 / sr
        scale_min     = fc / (fmax * dt)     # small scale → high frequency
        scale_max     = fc / (fmin * dt)     # large scale → low frequency
        self._scales  = np.geomspace(scale_min, scale_max, self.n_internal)

        # Output frequency grid (uniform Hz bins for the reassigned scalogram)
        self._out_freqs = np.linspace(fmin, fmax, n_scales)  # (n_scales,)
        self._df        = self._out_freqs[1] - self._out_freqs[0]

    def encode(self, audio: np.ndarray) -> np.ndarray:
        import pywt

        dt = 1.0 / self.sr
        N  = len(audio)

        # ── Step 1: complex CWT ───────────────────────────────
        # coeffs: (n_internal, N)  complex
        coeffs, _ = pywt.cwt(audio, self._scales, self._wavelet,
                              sampling_period=dt)
        coeffs = coeffs.astype(np.complex64)

        # ── Step 2: instantaneous frequency ──────────────────
        # Central finite difference along time axis
        dcoeffs          = np.empty_like(coeffs)
        dcoeffs[:, 1:-1] = (coeffs[:, 2:] - coeffs[:, :-2]) / (2.0 * dt)
        dcoeffs[:, 0]    = dcoeffs[:, 1]
        dcoeffs[:, -1]   = dcoeffs[:, -2]

        # Avoid division by near-zero coefficients
        magnitude  = np.abs(coeffs)                          # (n_internal, N)
        safe_denom = np.where(magnitude > 1e-10, coeffs, 1e-10 + 0j)
        inst_freq  = np.abs(np.imag(dcoeffs / safe_denom) / (2.0 * np.pi))
        inst_freq  = np.clip(inst_freq, self.fmin, self.fmax)  # (n_internal, N)

        # ── Step 3: synchrosqueezing (energy reassignment) ───
        # Map each (scale, time) energy to its instantaneous-frequency bin
        squeezed = np.zeros((self.n_scales, N), dtype=np.float32)
        bin_idx  = np.round(
            (inst_freq - self.fmin) / self._df
        ).astype(np.int32)
        bin_idx  = np.clip(bin_idx, 0, self.n_scales - 1)   # (n_internal, N)

        # Vectorised scatter-add using np.add.at
        for s in range(self.n_internal):
            np.add.at(squeezed, (bin_idx[s], np.arange(N)), magnitude[s])

        # ── Step 4: decimate and convert to dB ───────────────
        squeezed = squeezed[:, ::self.decimation]            # (n_scales, T_frames)
        squeezed = 20.0 * np.log10(squeezed + 1e-9)
        return squeezed                                      # (n_scales, T_frames)

    @property
    def n_channels(self) -> int:
        return self.n_scales

    def __repr__(self):
        return (f"ChirpletEncoder(sr={self.sr}, n_scales={self.n_scales}, "
                f"fmin={self.fmin}, fmax={self.fmax}, "
                f"decimation={self.decimation}, "
                f"n_internal={self.n_internal})")

