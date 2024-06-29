/*--------------------------------------------
    Likelihood-Evaluation of SWIFT III
    generative mode also works

    Note: To use multithreading, compile with
    -fopenmp for OpenMP. To explicitly disable
    multithreading, compile with -D DISABLE_THREADS.

 =========================================
 (Version 1.0, 26-AUG-02)    
 (Version 2.0, 23-JUN-03)
 (Version 3.0, 10-DEC-03)
 (Version 4.0, 25-JUN-04)
 (Version 4.1, 19-MAY-07)
 (Version 5.0, 07-OCT-11)
 ----------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <libgen.h>

#include "swiftstat7_api.c"

typedef enum {USAGE, HELP, VERSION, GENERATE, FITTING, CITATION, DEFAULTS, VALIDATE} swift_mode;

int win_width;

#if defined __MINGW32__
#define set_win_width(var, def) {var = def;}
#elif defined __GNUC__
#include <sys/ioctl.h>
#define set_win_width(var, def) {struct winsize w; w.ws_col = 0; ioctl(STDOUT_FILENO, TIOCGWINSZ, &w); if(w.ws_col > 0) win_width = w.ws_col; else win_width = def;}
#endif

// Some output functions

size_t ebstrlen(const char *str, size_t ub) {
    size_t i, ret;
    unsigned short skip = 0;
    for(i=0,ret=0;str[i]!=0&&(ub<0||i<ub);i++) {
        if(str[i]=='\e'&&str[i+1]=='[')
            skip = 1;
        if(skip && str[i] == 'm') {
            skip = 0;
            continue;
        }
        if(skip)
            continue;
        switch(str[i]) {
            case '\t': ret+=8; break;
            case '\n': break;
            case '\r': break;
            default: ret++; break;
        }
    }
    return ret;
}

size_t estrlen(const char *str) {
    return ebstrlen(str, -1);
}

void fwrap(FILE* out, unsigned int w, unsigned int lm, char *prefix, char *ofmt, ...) {
    char empty[1] = {0}, line[w+1], *content;
    unsigned int bufsize, buflen, bufcur = 0, linelen = 0, i, l = 0;
    for(bufsize=512;;bufsize+=512) {
        content = calloc(sizeof(char), bufsize);
        va_list argptr;
        va_start(argptr, ofmt);
        buflen = vsnprintf(content, bufsize, ofmt, argptr);
        va_end(argptr);
        if(buflen < bufsize) break;
    }
    if(w<lm+5) {
        if(prefix == NULL) {
            fprintf(out, "%s\n", content);
        }else{
            fprintf(out, "%s %s\n", prefix, content);
        }
        return;
    }
    if(prefix!=NULL && estrlen(prefix) > lm) {
        lm = estrlen(prefix);
    }
    while(bufcur<buflen) {
        if(isspace(content[bufcur])&&l>0) {
            bufcur++;
            continue;
        }
        for(i=0;ebstrlen(&content[bufcur],i)<w-lm;i++) {
            if(content[bufcur+i]=='\n'||content[bufcur+i]==0) {
                linelen = i;
                break;
            }
            if(content[bufcur+i]==' ') {
                linelen = i;
            }
        }
        if(linelen == 0) linelen = strlen(&content[bufcur]);
        memcpy(line, &content[bufcur], linelen);
        line[linelen] = 0;
        if(l == 0 && prefix != NULL) {
            fprintf(out, "%s%*.*s%s\n", prefix, estrlen(prefix)>lm ? 0 : (int) (lm-estrlen(prefix)), estrlen(prefix)>lm ? 0 : (int) (lm-estrlen(prefix)), " ", line);
        }else{
            fprintf(out, "%*s%.*s\n", lm, empty, w-lm, line);
        }
        bufcur+=linelen;
        l++;
    }
    free(content);
}

/**
	This method parses integer range string literals and writes each item
	into a target array. Note that this method does not take care of
	duplicates. The method always returns the number of items or -1 if an
	error occured while parsing.

	str: A string containing a comma-separated list of integers and/or
	     integer ranges (e.g., 1,3-6,9 for 1,3,4,5,6,9).
	target: A target array in which the individual items are stored in
	     the order in that they appear in str. You should make sure that
         the array has enough space for all items. If NULL, nothing items
         are not stored but only counted.

    Note: If you are unsure how many items are in str and thus how much
    memory you should allocate for the target array, you can run the
    method with target=NULL first to count the items, then declare or
    allocate the target array and run again with the same str and the
    actual target array. If doing so, make sure to catch the case that
    the return value is -1 which indicates a parsing error and would not
    be a valid array length!
**/
int parse_range_literal(char * str, int * target) {
	const int tmpbuflen = 5;
	char tmp[tmpbuflen+1];
	int tmplen = 0;
	int ptr = 0, strl = strlen(str);
	int from_item = -1, to_item = -1;
	int i;
	int n = 0;
	char tmpc;
	for(ptr = 0; ptr <= strl; ptr++) {
		// iterate through str string char by char
		if(str[ptr] == ' ') continue; // ignore whitespace
		if(str[ptr] == 0 || str[ptr] == ',') {
			// when comma is found or end of string reached, parse temp string into number (upper bound of current range)
			if(sscanf(tmp, "%d%c", &to_item, &tmpc) != 1) {
				error(1, "“%s” is not a valid integer.", tmp);
				return -1;
			}
			if(from_item == -1) {
				// if no lower bound was defined (i.e., this list entry is a single number, not a range), behave as if this is a single-value range
				from_item = to_item;
			} else if(from_item > to_item) {
				// throw an error if upper bound is less than lower bound
				error(1, "%d-%d is not a valid range. First number (lower bound) must be less than or equal to second number (upper bound).", from_item, to_item);
				return -1;
			}
			if(target != NULL) for(i = 0; i <= to_item-from_item+1; i++) {
				// add each number within the range to the output array, item by item
				// this only happens if target != NULL, thereby allowing dry runs which only count the number of items to be added
				target[n+i] = from_item+i;
			}
			n += to_item-from_item+1; // add number of items to total number of items added
			from_item = -1; // reset lower range bound
		} else if(str[ptr] == '-') {
			// when minus sign is found, the temp string is interpreted as the lower bound of a range
			if(from_item != -1) {
				// if a range has already been parsed and we find another minus sign before a comma or end of string, throw an error
				error(1, "Unexpected range format with more than one minus sign. A range must be defined by only two integers.");
				return -1;
			}
			if(sscanf(tmp, "%d%c", &from_item, &tmpc) != 1) {
				// try to parse the temp string as an integer and throw an error if it has an unknown format
				error(1, "“%s” is not a valid integer.", tmp);
				return -1;
			}
		} else {
			// if this is not a special character (comma, end of string, whitespace or minus), add the character to the temp string, which is evaluated later (see above)
			if(tmplen >= tmpbuflen) {
				// do not add anything to the buffer if it is already full!
				error(1, "Cannot parse integers longer than %d characters.", tmpbuflen);
				return -1;
			}
			tmp[tmplen++] = str[ptr];
			tmp[tmplen] = 0;
		}
		if(str[ptr] == ',' || str[ptr] == 0 || str[ptr] == '-') {
			// if this was a special character marking a number boundary (comma, end of string or minus), assume it has been properly interpreted and reset the temp string buffer
			tmp[0] = 0;
			tmplen = 0;
		}
	}
	// return the number of added items
	return n;
}

#ifdef DISABLE_THREADS
	static char usage_t_placeholder_short[] = "", usage_t_placeholder_long[] = "";
#else
	static char usage_t_placeholder_short[] = " [-t nthreads]", usage_t_placeholder_long[] = " [--threads nthreads]";
#endif

void print_usage(FILE* where, char* cmd_name) {
	fwrap(where, win_width, 0, cmd_name, " -uhvfgCd [-q] [-c cid [-p pid] [-P a=1.0,... [-P ...]] [-I snr] [-s sid] [-x] [-r seed] [-i input] [-e envdir] [-o output]%s]", usage_t_placeholder_short);
	fwrap(where, win_width, 0, cmd_name, " --[usage|help|version|fitting|generate|citation format|defaults] [--quiet] [--corpus cid [--parix pid] [--param a=1.0,... [--param b=2.0,...]] [--sentence snr] [--seqid sid] [--fixseq] [--ranseed seed] [--input inpath] [--environ envdir] [--output outpath]%s]", usage_t_placeholder_long);
}

struct help_item {
	int key;
	char *msg;
};



int main(int argc, char **argv) {

	int i, j, k, c;

	char version_string[512];
	swift_complete_version_string(version_string, NULL);


	// get default number of threads depending on thread model specified during compiling
	#ifndef DISABLE_THREADS
	const int default_num_threads = omp_get_max_threads();
	#else
	const int default_num_threads = 1;
	#endif

	// these are long argument option alternatives
	static struct option long_options[] = {
		/* These options don’t set a flag.
		 We distinguish them by their indices. */
		{"usage",   no_argument, 0, 'u'},
		{"help",   no_argument, 0, 'h'},
		{"version",   no_argument, 0, 'v'},
		{"citation",   required_argument, 0, 'C'},
		{"fitting",   no_argument, 0, 'f'},
		{"generate",   no_argument, 0, 'g'},
		{"defaults",   no_argument, 0, 'd'},
		{"validate",   no_argument, 0, 'V'},
		{"corpus",  required_argument, 0, 'c'},
		{"seqid",   required_argument, 0, 's'},
		{"parfile",   required_argument, 0, 'p'},
		{"parix",   required_argument, 0, 'p'},
		{"param",   required_argument, 0, 'P'},
		{"ranseed",   required_argument, 0, 'r'},
		{"rand",   required_argument, 0, 'r'},
		{"output",   required_argument, 0, 'o'},
		{"input",   required_argument, 0, 'i'},
		{"environ",   required_argument, 0, 'e'},
		{"quiet",   no_argument, 0, 'q'},
		{"fixseq",   no_argument, 0, 'x'},
		{"sentence",   required_argument, 0, 'I'},
		{"item",   required_argument, 0, 'I'},
		{"threads",   required_argument, 0, 't'},
		{0}
	};

	// -t/--threads helpmsg depends on threading model
	#ifndef DISABLE_THREADS
	static char threads_helpmsg[250];
	sprintf(threads_helpmsg, "How many subsets to evaluate in parallel (current default: %d). Only implemented for fitting mode. Note: As this as an OpenMP version, the default can be modified by setting the environment variable OMP_NUM_THREADS.", default_num_threads);
	#endif

	static char citation_helpmsg[500];
	char tmp[200] = {0};
	for(i=0;swift_citations[i].style!=NULL;i++) {
		sprintf(&tmp[strlen(tmp)], ", %s", swift_citations[i].style);
	}
	sprintf(citation_helpmsg, "Show citation for this version. Citation style to be passed as an additional argument (supported: all%s). If passed with an asterisk (*) after the citation style (e.g., all*), output is in Markdown format.", tmp);

	// these are help messages for CLI arguments
	static struct help_item help_items[] = {
		{'h', "Show help message. Other options ignored."},
		{'u', "Show command-line usage (short help). Other options ignored."},
		{'v', "Show current version. Other options ignored."},
		{'C', citation_helpmsg},
		{'f', "Enter fitting mode."},
		{'g', "Enter generative mode."},
		{'V', "Validate fixation sequences."},
		{'d', "Show parameter default values. If used with -o, default parameters are also written to the specified file."},
		{'e', "Read environment (swiftstat.inp) and corpus (corpus_*.dat) files from this directory (default: same as -i)."},
		{'r', "Set a random seed (long integer) for all generated random numbers."},
		{'p', "Read parameters from swpar_{cid}_{sid}_{pid}.par. If not set, read parameter from swpar_{cid}_{sid}.par."},
		{'P', "After loading parameters, update with these values. To update multiple parameters, either use this option several times (-P a=1.0 -P b=2.0), pass a comma- or semicolon-separated list (-P a=1.0,b=2.0) or both (-Pa=1.0 --param b=2.0;c=3.0)."},
		{'c', "Read corpus from corpus_{cid}.dat. Required for \e[1mgenerative\e[0m and \e[1mfitting\e[0m mode."},
		{'s', "Read fixation sequences from fixseqin_{sid}.dat."},
		{'o', "For \e[1mfitting mode\e[0m: Write log-likelihoods into this file (provide complete path). If not provided, output is written into console only. For \e[1mgenerative mode\e[0m: Write seq_{cid}_{sid}.dat and mis7_{cid}_{sid}.dat into this directory (default: ../SIM). When \e[1mdisplaying defaults\e[0m (-d): Write default parameter values into this file."},
		{'i', "Find fixation sequences (fixseqin_*.dat) and parameter (swpar_*.par) files in this directory (default: ../DATA)."},
		{'I', "For \e[1mgenerative mode\e[0m, only simulate the sentences specified by the corpus sentence number(s). For \e[1mfitting\e[0m mode, only evaluate trials specified by their trial number(s). Numbers are to be passed as an additional argument, e.g. “6”, “1,7“, or “5-18,27”."},
		{'x', "For \e[1mgenerative mode\e[0m, create a fixseq_*.dat file in addition to generated output (seq_*.dat)."},
		#ifndef DISABLE_THREADS
		{'t', threads_helpmsg},
		#endif
		{0}
	};

	set_win_width(win_width, 80);

    // set default values
	unsigned short verbose = 1, seq2fixseq = 0;
	swift_mode mode = USAGE;

	// get default random seed
	uint64_t super_seed = 0l;
	FILE *ranfile = fopen("/dev/random", "r");
	if(ranfile == NULL) {
		// could not open /dev/random --> fallback to epoch time + process id + parent process id
		struct timeval t;
		gettimeofday(&t, NULL);
		super_seed = ((long)t.tv_sec*1000l) + ((long)t.tv_usec) + getpid() * t.tv_usec;
	}else{
		fread(&super_seed, sizeof(uint64_t), 1, ranfile);
		fclose(ranfile);
	}

	// set default number of threads as default number for requested number of threads
	int num_threads = default_num_threads;

	// default values for these arguments will be set after argument parsing -> NULL marks no value has been set
	char *corpus_id = NULL, *fixseq_id = NULL, *par_id = NULL, *environ_path = NULL, *inpath = NULL, *outpath = NULL, *citation_style=NULL;

	int * items = NULL;
	int n;

	swift_parameters *updates = alloc_parameters(NULL);

		while(1) {
		int option_index = 0;
		c = getopt_long(argc, argv, "dufghvVC:c:s:p:t:o:i:r:P:qxe:I:", long_options, &option_index);
		if(c == -1) {
			break;
		}

		char dummy;

		switch(c) {
			// modes
			case 'u':
				mode = USAGE;
				break;
			case 'd':
				mode = DEFAULTS;
				break;
			case 'h':
				mode = HELP;
				break;
			case 'f':
				mode = FITTING;
				break;
			case 'g':
				mode = GENERATE;
				break;
			case 'V':
				mode = VALIDATE;
				break;
			case 'v':
				mode = VERSION;
				break;
			case 'C':
				mode = CITATION;
				if(strcmp(optarg,"all")) // if argument is "all", leave NULL
					citation_style = optarg;
				break;
			// options
			case 'c':
				corpus_id = optarg;
				break;
			case 's':
				fixseq_id = optarg;
				break;
			case 'p':
				par_id = optarg;
			case 'q':
				verbose = 0;
				break;
			case 'x':
				seq2fixseq = 1;
				break;
			case 'I':
				n = parse_range_literal(optarg, NULL);
				if(n < 1) {
					stop(1, "Invalid sentence number range literal “%s”. See error messages for details.", optarg);
				}
				items = malloc(sizeof(int) * (n+1));
				parse_range_literal(optarg, items);
				items[n] = 0;
				break;
			case 'P':
				{
					int i = -1, j;
					char parname[50], tmp[50];
					double ddum;
					int idum;
					char *token = optarg;
					do {
						memset(tmp, 0, strlen(tmp)+1);
						j = 0;
						for(i++;token[i]!=0&&token[i]!='=';i++) {
							if(token[i]!=' '&&token[i]!='\t') {
								tmp[j++] = token[i];
							}				
						}
						tmp[j++] = 0;
						if(token[i]!='=') {
							stop(1, "Unrecognized parameter token '%s'! Please use parone=1.0,partwo=2.0,...", token);
						}
						sprintf(parname, "%s", tmp);
						// is this a valid model parameter?
						int par_id, par_type = 0;
						// check if this is a valid model parameter:
						par_id = swift_find_param(parname);
						if(par_id < 0) {
							stop(1, "Unrecognized parameter name '%s'!\n", parname);
						}
						memset(tmp, 0, strlen(tmp)+1);
						j = 0;
						for(i++;token[i]!=0&&token[i]!=','&&token[i]!=';';i++) {
							if(token[i]!=' '&&token[i]!='\t') {
								tmp[j++] = token[i];
							}
						}
						tmp[j++] = 0;
						if(token[i]!=','&&token[i]!=';'&&token[i]!=0) {
							stop(1, "Unrecognized character '%c'! Please use parone=1.0,partwo=2.0,...", token[i]);
						}
						if(!scan_parameter(tmp, updates, par_id)) {
							stop(1, "Could not read parameter value for “%s” from command line!", parname);
						}
					} while(token[i]!=0);
				}
				break;
			case 'r':
				super_seed = 0l;
				if(sscanf(optarg, "%lu%c", (unsigned long int*) &super_seed, &dummy) != 1) {
					stop(1, "Seed provided (%s) must be an unsigned integer (0-%lu)!", optarg, RANSEED_MAXSEED);
				}
				break;
			case 'o':
				outpath = optarg;
				break;
			case 'i':
				inpath = optarg;
				break;
			case 'e':
				environ_path = optarg;
				break;
			case 't':
				if(sscanf(optarg, "%d%1s", &num_threads, &dummy) != 1 || num_threads < 1) {
					stop(1, "Number of threads provided (%s) is not an integer greater than 0!", optarg);
				}
				#ifdef DISABLE_THREADS
				if(num_threads > 1) warn("You requested multiple threads (using command option -t or --threads), which is not enabled and will thus be ignored. If you need to use multithreading capabilities, compile with OpenMP.");
				#endif
				break;
			default:
				warn("Unknown argument! Ignored.");
				break;
		}
	}

	if(mode == USAGE) {
		if(corpus_id != NULL || fixseq_id != NULL) {
			warn("You specified a corpus identifier (using -c or --corpus) and/or a fixation sequence identifier (using -s or --seqid). In addition, please select fitting (using -f or --fitting), generative (using -g or --generate) or validation (using -V or --validate) mode or see help (using -h or --help) to explore available options!");
			print_usage(stderr, basename(argv[0]));
			return 1;
		}else {
			print_usage(stderr, basename(argv[0]));
			return 0;
		}
	}else if(mode == HELP) {

		char cmd_prefix[strlen(argv[0])+5];
		sprintf(cmd_prefix, "   %s", basename(argv[0]));

		//fcenterp(stderr, win_width, '=', 2, 2, "\e[1m%s\e[0m", version_string);
		fprintf(stderr, " == \e[1m%s\e[0m ==", version_string);

		fprintf(stderr, "\n\e[1m\e[4mUsage:\e[0m\n");
		print_usage(stderr, cmd_prefix);

		unsigned int arglen = 0;
		for(i=0;long_options[i].name!=0;i++)
			if(strlen(long_options[i].name)>arglen)
				arglen=strlen(long_options[i].name);

		fprintf(stderr, "\n\e[1m\e[4mCommand line arguments:\e[0m\n");
		for(i=0;help_items[i].key!=0;i++) {
			char prefix[30];
			for(j=0;long_options[j].name!=0;j++)
				if(long_options[j].val==help_items[i].key)
					break;
			if(long_options[j].name!=0) {
				sprintf(prefix, "   \e[1m-%c --%s\e[0m", help_items[i].key, long_options[j].name);
				fwrap(stderr, win_width, arglen+10, prefix, help_items[i].msg);
				for(k=j+1;long_options[k].name!=0;k++)
					if(long_options[k].val==help_items[i].key) {
						sprintf(prefix, "      \e[1m--%s\e[0m", long_options[k].name);
						fwrap(stderr, win_width, arglen+10, prefix, "(alias for \e[1m--%s\e[0m)", long_options[j].name);
					}
			}
			else {
				sprintf(prefix, "   \e[1m-%c\e[0m", help_items[i].key);
				fwrap(stderr, win_width, 18, prefix, help_items[i].msg);
			}
		}

		fprintf(stderr, "\n\e[1m\e[4mExamples:\e[0m\n");

		fprintf(stderr, "\nTo show this help message:\n");
		fwrap(stderr, win_width, 0, cmd_prefix, " --help");
		fwrap(stderr, win_width, 0, cmd_prefix, " -h");
		fprintf(stderr, "\nTo evaluate the log-likelihood of a dataset given a set of parameters:\n");
		fwrap(stderr, win_width, 0, cmd_prefix, " --fitting --corpus cid --seqid sid [--parix pid] [--param a=1.0,b=2.0,...] [--ranseed seed] [--input inpath] [--environ envdir] [--output outfile]%s", usage_t_placeholder_long);
		fwrap(stderr, win_width, 0, cmd_prefix, " -f -c cid -s sid [-p pid] [-P a=1.0,b=2.0,...] [-r seed] [-i inpath] [-e envdir] [-o outfile]%s", usage_t_placeholder_short);
		fprintf(stderr, "\nTo simulate a dataset given a corpus (cid) and a set of parameters:\n");
		fwrap(stderr, win_width, 0, cmd_prefix, " --generate --corpus cid --seqid sid [--parix pid] [--sentence snr] [--param a=1.0,b=2.0,...] [--ranseed seed] [--fixseq] [--input inpath] [--environ envdir] [--output outpath]");
		fwrap(stderr, win_width, 0, cmd_prefix, " -g -c cid -s sid [-p pid] [-I snr] [-P a=1.0,b=2.0,...] [-r seed] [-x] [-i inpath] [-e envdir] [-o directory]");
		return 0;
	}else if(mode == VERSION) {
		printf("%s\n", version_string);
		return 0;
	}else if(mode == CITATION) {
		unsigned int markdown = 0;
		if(citation_style != NULL && citation_style[strlen(citation_style)-1] == '*') {
			markdown = 1;
			citation_style[strlen(citation_style)-1] = 0;
		}
		if(citation_style == NULL || !strcmp(citation_style, "all")) {
			for(i=0;swift_citations[i].style!=NULL;i++) {
				char *ret = NULL;
				if(markdown) {
					ret = swift_citations[i].markdown;
				} else {
					ret = swift_citations[i].plain;
				}
				if(ret!= NULL)
					fprintf(stdout, "\e[2m%s\e[22m: %s\n", swift_citations[i].style, ret);
			}
			return 0;
		} else {
			swift_citation *swift_cite = swift_find_citation_style(citation_style);
			if(swift_cite != NULL) {
				char *ret = NULL;
				if(markdown) {
					ret = swift_cite->markdown;
				} else {
					ret = swift_cite->plain;
				}
				if(ret == NULL) {
					stop(1, "The citation style '%s' is not available in the requested format!", citation_style);
				}else{
					fprintf(stdout, "%s\n", ret);
					return 0;
				}
			}
			stop(1, "Unknown citation format '%s'! Please see help (%s -h) for supported citation formats!", citation_style, basename(argv[0]));
		}
	}else if(mode == FITTING || mode == GENERATE || mode == VALIDATE) {
		char corpus_file[PATH_MAX], fixseq_file[PATH_MAX], par_file[PATH_MAX];
		if(inpath  == NULL) 
			inpath = "../DATA";

		if(environ_path == NULL) 
			environ_path = inpath;

		if(corpus_id == NULL)
			stop(1, "Corpus identifier must be specified (using -c or --corpus, see --help)!");
		
		sprintf(corpus_file, "%s/corpus_%s.dat", environ_path, corpus_id);
		if(fixseq_id == NULL)
			stop(1, "Fixation sequence identifier must be specified (using -s or --seqid, see --help)!");
		
		sprintf(fixseq_file, "%s/fixseqin_%s.dat", inpath, fixseq_id);
		if(par_id == NULL) {
			sprintf(par_file, "%s/swpar_%s_%s.par", inpath, corpus_id, fixseq_id);
		}else{
			sprintf(par_file, "%s/swpar_%s_%s_%s.par", inpath, corpus_id, fixseq_id, par_id);
		}

		if(verbose) {
			//fcenterp(stdout, win_width, '=', 2, 2, "\e[1m%s\e[0m", version_string);
			fprintf(stderr, " == \e[1m%s\e[0m ==", version_string);
			printf("\n\tRandom seed: %llu\n\n", super_seed);
		}

		swift_model *model;

		int errcode;

		// Load model

		if(!swift_load_model(environ_path, par_file, corpus_file, super_seed, &model, verbose)) {
			stop(1, "Loading model failed.");
		}

		if(verbose) {
			printf("\n\t1. Simulation and model parameters:\n");
			print_parameters(stdout, model->params);
			printf("\n\t2. Corpus data:\n");
			printf("\t   %d sentences\n", nsentences(model->corpus));
			printf("\n\t3. Update parameters through console:\n");
			print_parameters(stdout, updates);
		}

		// Update model parameters
		int has_updated_parameters = 0;
		for(i=0;swift_parameters_meta[i].name!=NULL;i++) {
			if(swift_parameters_meta[i].type == PARTYPE_INTEGER && hasvalbyid(updates, i, swift_parameter_int)) {
				setvalbyid(model->params, i, swift_parameter_int, valbyid(updates, i, swift_parameter_int));
				has_updated_parameters = 1;
			} else if(swift_parameters_meta[i].type == PARTYPE_DOUBLE && hasvalbyid(updates, i, swift_parameter_dbl)) {
				setvalbyid(model->params, i, swift_parameter_dbl, valbyid(updates, i, swift_parameter_dbl));
				has_updated_parameters = 1;
			}
		}

		if(verbose) {
			if(!has_updated_parameters) printf("\t(none)\n");
			else {
				printf("\n\t   After updating:\n");
				print_parameters(stdout, model->params);
			}
		}

		if(mode == VALIDATE) {
			swift_dataset *cdata;

			if(!swift_load_data(fixseq_file, &cdata, verbose)) {
				stop(1, "Loading fixation sequences failed.");
			}

			if(verbose) {
				printf("\n\t4. Empirical data:\n");
				printf(  "\t   %d fixation sequences\n",ntrials(cdata));
			}

			if(swift_validate(cdata, model)) {
				free_swift_dataset(cdata);			
				free_swift_model(model);
				printf("\n\n\tFixation sequences are valid.\n");
				exit(0);
			}else{
				free_swift_dataset(cdata);			
				free_swift_model(model);
				printf("\n\n\tFixation sequences are INVALID!\n");
				exit(1);
			}


		}else if(mode == FITTING) {
			swift_dataset *cdata;

			if(!swift_load_data(fixseq_file, &cdata, verbose)) {
				stop(1, "Loading fixation sequences failed.");
			}

			if(verbose) {
				printf("\n\t4. Fitting empirical data:\n");
				printf(  "\t   %d fixation sequences\n",ntrials(cdata));
			}

			if(!swift_validate(cdata, model)) {
				free_swift_dataset(cdata);			
				free_swift_model(model);
				stop(1, "There was an error during sequence validation. Fitting will be aborted due to corrupt data.");
			}


			if(items!=NULL) {
				for(i=0;i<n;i++) {
					if(items[i]<=0||items[i]>ntrials(cdata)) {
						stop(2, "You requested trial %d but the sequence file only contains %d trial(s).", items[i], ntrials(cdata));
					}
				}
			}

			double retvals[N_LOGLIKS]; // if no outputfile specified, get them here

			swift_eval(model, cdata, items, retvals, num_threads, verbose);
			free_swift_dataset(cdata);

			for(i=0;i<N_LOGLIKS;i++) {
				if(i>0&&verbose) printf(", ");
				else if(i>0) printf(" ");
				else if(verbose) printf("\n\tloglik: ");
				printf("%lf", retvals[i]);
				if(i==N_LOGLIKS-1) printf("\n");
			}
			

			free_swift_model(model);

		}else if(mode == GENERATE) {

			if(outpath == NULL)
				outpath = "../SIM";

			if(items!=NULL) 
				for(i=0;i<n;i++) {
					if(items[i]<=0||items[i]>nsentences(model->corpus)) {
						stop(2, "The sentence/item requested (%d) was not found in corpus %s! Valid values for this corpus are 1..%d.", items[i], corpus_id, nsentences(model->corpus));
					}
				}

			swift_generate(model, outpath, fixseq_id, items, num_threads, seq2fixseq, verbose);

			free_swift_model(model);

		}

		return 0;
		
	} else if(mode == DEFAULTS) {
		swift_parameters *pars = alloc_parameters(NULL);
		setdefaults(pars);
		if(verbose) print_parameters(stdout, pars);
		if(outpath != NULL) {
			FILE *f = fopen(outpath, "a");
			if(f == NULL) {
				stop(1, "Could not open “%s” for writing default parameter values.", outpath);
			}
			write_parameters(f, pars);
			fclose(f);
		}
		free_parameters(pars);
		return 0;
	}

	if(items != NULL) free(items);


	warn("There was nothing to do.");

	return 1;
	
}
