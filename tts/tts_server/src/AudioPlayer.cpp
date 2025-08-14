#include "AudioPlayer.h"
#include <iostream>

AudioPlayer::AudioPlayer()
{
    initialize();
}

AudioPlayer::~AudioPlayer()
{
    cleanup();
}

bool AudioPlayer::initialize()
{
    if (initialized_)
        return true;

    int err = snd_pcm_open(&pcm_handle_, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0)
    {
        std::cerr << "ALSA Error: Cannot open PCM device: " << snd_strerror(err) << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

void AudioPlayer::play(const int16_t *audioData, int audio_len, float speed)
{
    if (!initialized_ || !pcm_handle_)
        return;

    unsigned int sample_rate = static_cast<unsigned int>(16000 * speed);
    int err = snd_pcm_set_params(pcm_handle_,
                                 SND_PCM_FORMAT_S16_LE,
                                 SND_PCM_ACCESS_RW_INTERLEAVED,
                                 1,
                                 sample_rate,
                                 1,
                                 50000);
    if (err < 0)
    {
        std::cerr << "ALSA Error: Cannot set parameters: " << snd_strerror(err) << std::endl;
        return;
    }

    const snd_pcm_uframes_t frames = audio_len / 2;
    const int max_retries = 3;
    int retry_count = 0;

    while (true)
    {
        err = snd_pcm_writei(pcm_handle_, audioData, frames);
        if (err == -EPIPE)
        {
            if (++retry_count >= max_retries)
                break;
            snd_pcm_prepare(pcm_handle_);
        }
        else if (err < 0)
        {
            break;
        }
        else
        {
            break;
        }
    }

    if (snd_pcm_state(pcm_handle_) == SND_PCM_STATE_RUNNING)
    {
        snd_pcm_drain(pcm_handle_);
    }
}

void AudioPlayer::cleanup()
{
    if (pcm_handle_)
    {
        snd_pcm_close(pcm_handle_);
        pcm_handle_ = nullptr;
    }
    initialized_ = false;
}