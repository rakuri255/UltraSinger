import wave
import json
import csv

from vosk import Model, KaldiRecognizer
from moduls.Speech_Recognition.VoskTranscribedData import VoskTranscribedData
from moduls.Audio.vocal_chunks import export_chunks_from_vosk_data

# todo: Rename to Transcoder?

def write_lists_to_csv(list_of_vosk_words, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["word", "start", "end", "confidence"]
        writer.writerow(header)
        for i in range(len(list_of_vosk_words)):
            writer.writerow(
                [list_of_vosk_words[i].word, list_of_vosk_words[i].start, list_of_vosk_words[i].end, list_of_vosk_words[i].conf])


def transcribe_with_vosk(audio_filename, folder_name, model_path):
    # Code from here: https://towardsdatascience.com/speech-recognition-with-timestamps-934ede4234b2
    print("Transcribing {} with vosk and model {}".format(audio_filename, model_path))

    csv_filename = folder_name + "/_chunks.csv"

    model = Model(model_path)
    wf = wave.open(audio_filename, "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)

    # get the list of JSON dictionaries
    results = []
    # recognize speech using vosk model
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            part_result = json.loads(recognizer.Result())
            results.append(part_result)
    part_result = json.loads(recognizer.FinalResult())
    results.append(part_result)

    # convert list of JSON dictionaries to list of 'Word' objects
    list_of_words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for obj in sentence['result']:
            w = VoskTranscribedData(obj)  # create custom Word object
            list_of_words.append(w)  # and add it to list

    # output to the screen
    #todo: progress?
    #for word in list_of_words:
    #    print(word.to_string())

    # todo: to own function
    export_chunks_from_vosk_data(wf, list_of_words, folder_name)
    write_lists_to_csv(list_of_words, csv_filename)

    wf.close()  # close audiofile

    return list_of_words


class SpeechToText:
    pass
