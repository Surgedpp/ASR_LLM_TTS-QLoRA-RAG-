#ifndef AUDIO_PLAYER_H
#define AUDIO_PLAYER_H

#include <cstdint>
#include <alsa/asoundlib.h>

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    
    bool initialize();
    void play(const int16_t* audioData, int audio_len, float speed = 1.0f);
    
private:
    void cleanup();
    
    snd_pcm_t* pcm_handle_ = nullptr;
    bool initialized_ = false;
};

#endif // AUDIO_PLAYER_H