import whisper
import json
import os

model = whisper.load_model("base")
Audios = os.listdir("Audios")

for Audio in Audios:
    if("_" in Audio):
        number = Audio.split("_")[0]
        title = Audio.split("_")[1][:-4]
        print(number, title)
        result = model.transcribe(audio = f"Audios/{Audio}",
                                  language = "en",
                                  task = "translate",
                                  word_timestamps = False)

        chunks = []
        for segment in result["segments"]:
            chunks.append({"number" : number, "title" : title, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})

        chunks_with_metadata = {"chunks" : chunks, "text" : result["text"]}

        with open(f"jsons/{Audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

      