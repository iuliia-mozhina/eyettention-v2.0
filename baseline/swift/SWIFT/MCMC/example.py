import swiftstat7 as swift
import numpy as np


def load_model(corpus, path="../DATA", seed=None):
    return swift.loadmodel(outputPath=path, parmPath="%s/swpar_default.par" % path,
                           corpusFile="%s/corpus_%s.dat" % (path, corpus), seed=np.random.randint(0, 2 ** 48)
        if seed is None else seed)


def generate(model=0, filename=""):
    swift.generate(filename, model)


print(swift.version())

corpus = "celer"
path = "../../DATA18"
seed = 1234

model = load_model(corpus, path, seed)

# Generate output for each trial
filename = "generation"
print("Generating...")
generate(filename=filename, model=model)

# Free model
swift.freemodel(model)

