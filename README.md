# UltraSinger

_This project is currently only a proof of concept and is still under development.!_

_The output files are currently not really usable!_

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music. 
It also can re-pitch current UltraStar files.

Multiple AI models are used to extract text from the voice and to determine the pitch. 

## How to use

_Not all options working now!_
```commandline
    this.py [opt] [mode] [rec model] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    -a      Is default
            Creates all
    -u      Create ultrastar txt file
    -m      Create midi file
    -s      Create sheet file
    
    ## INPUT is ultrastar.txt ##
    -a    Is default
            Creates all
    -r      repitch Ultrastar.txt (input has to be audio)
    -p    Check pitch of Ultrastar.txt input
    -m      Create midi file

    [rec model]
    -v     vosk model path
      
    [extra]
    -k      Keep audio chunks
    
    [pitcher]
    -crepe  default
```

### Input Audio

For audio it uses Vosk transcription model. This model is not included. You can download it here [Link](https://alphacephei.com/vosk/models).
Make sure you take the right language. Also consider that a bigger model is more accurate, but also takes longer to transcribe.

```commandline
-i "input/music.mp3" -v "models\vosk-model-en-us-0.42-gigaspeech"
```

### Input UltraStar

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```