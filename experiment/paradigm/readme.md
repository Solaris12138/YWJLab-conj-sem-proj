# PsychoPy version
PyschoPy-2022.2.3

## Cautions
- We did not use PychoPy-2024.2.4, since it would speed the audio file up, even when we have already set the sampling frequency as 44.1kHz.
- It seems to be a problem caused by the audio-play backend. Newer version of PsychoPy  includes the backend 'ptb' rather than 'pygame'.