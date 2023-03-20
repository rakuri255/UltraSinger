[![](https://visitcount.itsvg.in/api?id=UltraSinger&label=Profile%20Views&color=0&icon=7&pretty=false)](https://visitcount.itsvg.in)

# UltraSinger 

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

_This project is still under development!_

_The output files from the full automatic are currently not really usable!
But it is usable for re-pitching ultrastar files._

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music. 
It also can re-pitch current UltraStar files and calculates the possible in-game score.

Multiple AI models are used to extract text from the voice and to determine the pitch. 

## Requirement

You need FFmpeg installed.

## How to use

_Not all options working now!_
```commandline
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default  Creates all
            
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all

    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file) # In Progress

    [transcription]
    --whisper   (default) tiny|base|small|medium|large
    --vosk      Needs model
    
    [extra]
    (-k                     Keep audio chunks) # In Progress
    --hyphenation           (default) True|False
    --disable_separation    True|False
    --disable_karaoke       True|False
    
    [pitcher]
    --crepe  (default) tiny|small|medium|large|full
```

### Input

#### Audio (full automatic)

##### Local file

```commandline
-i "input/music.mp3"
```

##### Youtube

```commandline
-i https://www.youtube.com/watch?v=BaW_jenozKc
```

#### UltraStar (re-pitch)

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### Transcriber

For transcription, `whisper` is used by default. It is more accurate than the other even with the `tiny` model.
And it finds the language automatically.
But anyway, it depends! Try the other one if `Whisper` does not suit you.
Also keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

#### Whisper

For the first test run, use the `tiny`, to be accurate use the `large` model

```commandline
-i XYZ --whisper large
```

#### Vosk

If you want to use `Vosk` than you need the model. It is not included. You can download it here [Link](https://alphacephei.com/vosk/models).
Make sure you take the right language. 
For the first test run, use the `small model`, to be accurate use the `gigaspeech` model

```commandline
-i "input/music.mp3" -v "models\vosk-model-en-us-0.42-gigaspeech"
```

#### Hyphenation

Is on by default. Can also be deactivated if hyphenation does not produce 
anything useful. Note that the word is simply split, 
without paying attention to whether the separated word really 
starts at the place or is heard.  

```commandline
-i XYZ --hyphenation True
```

### Pitcher

Pitching is done with the `crepe` model. 
Also consider that a bigger model is more accurate, but also takes longer to pitch.
For just testing you should use `tiny`, which is currently default.
If you want solid accurate, then use the `full` model.

```commandline
-i XYZ --crepe full
```

### Separation

The vocals are separated from the audio before they are passed to the models. If problems occur with this, 
you have the option to disable this function and the original audio file is used instead.

```commandline
-i XYZ --disable_separation True
```

### Ultrastar Score Calculation

The score what the singer in the audio would receive will be measured. 
You get 2 scores, simple and accurate. You wonder where the difference is? 
Ultrastar is not interested in pitch hights. As long as it is in the pitch range A-G you get one point. 
This makes sense for the game, because otherwise men don't get points for high female voices and women don't get points 
for low male voices. Accurate is the real tone specified in the txt. I had txt files where the pitch was in a range not 
singable by humans, but you could still reach the 10k points in the game. The accuracy is important here, because from
this MIDI and sheet are created. And you also want to have accurate files
