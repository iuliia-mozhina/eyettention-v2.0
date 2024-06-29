
import swiftstat7 as swift
import numpy as np

def load_model(corpus, seqid, path = "../DATA", seed = None):
	return swift.loadmodel(outputPath = path, parmPath = "%s/swpar_%s_%s.par" % (path, corpus, seqid), corpusFile = "%s/corpus_%s.dat" % (path, corpus), seed = np.random.randint(0,2**48) if seed is None else seed)

def load_data(seqid, path = "../DATA"):
	return swift.loaddata(fixseqName = "%s/fixseqin_%s.dat" % (path, seqid))

def update_parameter(param_name, param_value, model = 0):
	if type(param_name) is list:
		for i in range(len(param_name)):
			update_parameter(param_name[i], param_value[i], model)
	else:
		swift.updatemodel(param = param_name, value = param_value, model = model)

def get_parameter(param_name, model = 0):
	return swift.getmodel(model = model, field = param_name)

def loglik(model=0, data=0, threads = 0):
	return swift.eval(model = model, data = data, threads = threads, validate = False)

def validate(model=0, data=0):
	return swift.validate(model = model, data = data)
	
def generate(model=0, filename = ""):
	swift.generate(filename, model)

print((swift.version()))

model = load_model("oculo_01", "vp1_1_train", "../../DATA18", seed = 1234)

data = load_data("vp1_1_train", "../../DATA18")

validate(model, data)

update_parameter(["msac", "delta0"], [1.2, 8.77], model = model)

print((get_parameter("delta0", model)))

print((loglik(model, data))) # by default all available cores, add threads=N to set number of threads

generate(filename = "generation",model = model)

swift.freemodel(model)
swift.freedata(data)
