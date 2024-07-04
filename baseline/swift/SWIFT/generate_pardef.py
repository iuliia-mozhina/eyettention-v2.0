

# This script is used to generate a parameter definition file
# to be used when compiling the C code. It parses swpar.cfg (or
# any file passed as the first command-line argument) and
# produces structured C code. The result is written to standard
# output, from where it can be redirected to a file.


import re, sys

if len(sys.argv) < 2:
	print('Usage: '+sys.argv[0]+' file.cfg')
	exit()

splitter = re.compile("[ \t]+")

parnames = []
partypes = []
lines1 = {}
lines2 = []
lines3 = []

with open(sys.argv[1], "r") as f:
	for l in f.readlines():
		info = splitter.split(l.strip())
		if len(info) < 2:
			print("Invalid format of parameter definition!")
			exit(1)
		dtype = {'i':'int','d':'double','s':'char*'}[info[1]]
		dtype2 = {'i':'PARTYPE_INTEGER','d':'PARTYPE_DOUBLE','s':'PARTYPE_STRING'}[info[1]]
		dtype3 = {'i':'swift_parameter_int','d':'swift_parameter_dbl','s':'swift_parameter_str'}[info[1]]
		parnames.append(info[0])
		partypes.append(dtype2)
		if dtype3 not in lines1:
			lines1[dtype3] = []
		lines1[dtype3].append(info[0])
		if len(info) >= 3:
			lines3.append("setval(params, {parname}, {defval});".format(parname=info[0], defval=info[2]))

print("typedef struct {")
for t in lines1:
	print("\t"+t+" "+", ".join(lines1[t])+";")
print("} swift_parameters;")

print("#define setdefaults(params) {"+" ".join(lines3)+"}")

print("typedef enum {")
print("\t"+", ".join(["par_"+x for x in parnames]))
print("} swift_parameter_id;")

print("struct {")
print("\t swift_parameter_id "+", ".join(parnames)+";")
print("} swift_parameter_ids = {"+", ".join(["par_"+x for x in parnames])+"};")


print("swift_parameter_meta swift_parameters_meta[] = {")
for i in range(len(parnames)):
	print("\t{\""+parnames[i]+"\", "+partypes[i]+", par_"+parnames[i]+"},")
print("\t{0, 0, 0}")
print("};")

print("void* swift_param_addr(swift_parameters* pars, swift_parameter_id id) {")
print("\tswitch(id) {")
for p in parnames:
	print("\t\tcase par_"+p+": return &pars->"+p+";")
print("\t}")
print("\tstop(1, \"Unknown parameter id!\");")
print("\treturn NULL;")
print("}")