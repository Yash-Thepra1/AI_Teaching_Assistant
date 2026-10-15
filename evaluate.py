# Run from the project folder (same place as process_incoming.py) after embeddings.joblib exists.
# Needs Ollama running with the bge-m3 model, same as the main project.
import joblib, requests, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = joblib.load("embeddings.joblib")
matrix = np.vstack(df["embedding"])

def embed(text):
    r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": [text]})
    return r.json()["embeddings"][0]

# Harder test set: questions are paraphrased, not copied from the transcripts.
tests = [
    # video 1: CSS animations
    ("show me a quick way to animate something when I hover over it", "1"),
    ("how do I make a box move across the screen with css", "1"),
    ("what does position relative do for an effect", "1"),
    ("how do I make an animation change speed smoothly", "1"),
    # video 2: media in HTML
    ("how do I put a picture next to my list of songs", "2"),
    ("how can screen readers or search engines understand a picture", "2"),
    ("how do I draw shapes like rectangles inside a web page", "2"),
    # video 3: basic HTML
    ("how do I make text slanted", "3"),
    ("how do I make a list inside another list", "3"),
    ("what tags do I use for a grid of rows and columns", "3"),
    ("how do I add a heading to my page", "3"),
    # video 4: D3 intro
    ("what background do I need before learning this visualization library", "4"),
    ("what will the first lesson of the series cover", "4"),
    ("why is javascript needed for making charts in the browser", "4"),
    # video 5: D3 select and append
    ("how do I find an element on the page using code", "5"),
    ("how do I add a new paragraph to the page with javascript", "5"),
    ("how can I look at what my code did in the browser developer tools", "5"),
    ("how do I chain methods to change an element after selecting it", "5"),
    # video 6: D3 bind data
    ("how do I connect an array of numbers to shapes on screen", "6"),
    ("how do I make bars with different heights from my numbers", "6"),
    ("what is a placeholder selection before the data is attached", "6"),
    ("why are my bars drawn on top of each other", "6"),
    # video 7: HTML explained
    ("what is the difference between a tag and an attribute", "7"),
    ("what is the basic structure every web page has", "7"),
    ("explain html using a real life analogy", "7"),
    # video 8: Wikimedia Commons tools
    ("how do I move many files between categories at once", "8"),
    ("how do I turn on extra tools in my wikimedia account", "8"),
    ("which tools help people editing a media repository", "8"),
    # video 9: inline styles
    ("how do I put space around a box so it does not touch the edge", "9"),
    ("how do I change the look of one element without a separate stylesheet", "9"),
]

top1 = top5 = 0
for q, correct in tests:
    sims = cosine_similarity(matrix, [embed(q)]).flatten()
    idx = sims.argsort()[::-1][:5]
    nums = [str(int(df.loc[i, "number"])) for i in idx]
    hit1 = nums[0] == correct
    hit5 = correct in nums
    top1 += hit1; top5 += hit5
    print(("OK  " if hit5 else "MISS"), f"top1={'Y' if hit1 else 'N'}", correct, nums, "|", q)

n = len(tests)
print(f"\nTop-1 accuracy: {top1}/{n} = {100*top1/n:.0f}%")
print(f"Top-5 accuracy: {top5}/{n} = {100*top5/n:.0f}%")
