// Note: This is a temporary file and will be overwritten when the ./build.sh script is called.
// It is recommended to edit swpar.cfg instead, from which ./SIM/tmp/tmp_pardef.c file is generated.
typedef struct {
	swift_parameter_dbl misprob, delta0, delta1, asym, eta, alpha, beta, gamma, minact, theta, msac0, msac, h, h1, ppf, iota, refix, misfac, kappa0, kappa1, proc, decay, tau_l, tau_n, tau_ex, aord, cord, lord, nord, xord, ocshift, omn_fs1, omn_fs2, omn_sk1, omn_sk2, omn_frf1, omn_frf2, omn_brf1, omn_brf2, omn_rg1, omn_rg2, sre_fs1, sre_fs2, sre_sk1, sre_sk2, sre_frf1, sre_frf2, sre_brf1, sre_brf2, sre_rg1, sre_rg2;
	swift_parameter_int runs, nsims, output_ahist;
} swift_parameters;
#define setdefaults(params) {setval(params, misprob, 1.0); setval(params, runs, 1); setval(params, nsims, 300); setval(params, output_ahist, 0);}
typedef enum {
	par_misprob, par_runs, par_nsims, par_delta0, par_delta1, par_asym, par_eta, par_alpha, par_beta, par_gamma, par_minact, par_theta, par_msac0, par_msac, par_h, par_h1, par_ppf, par_iota, par_refix, par_misfac, par_kappa0, par_kappa1, par_proc, par_decay, par_tau_l, par_tau_n, par_tau_ex, par_aord, par_cord, par_lord, par_nord, par_xord, par_ocshift, par_omn_fs1, par_omn_fs2, par_omn_sk1, par_omn_sk2, par_omn_frf1, par_omn_frf2, par_omn_brf1, par_omn_brf2, par_omn_rg1, par_omn_rg2, par_sre_fs1, par_sre_fs2, par_sre_sk1, par_sre_sk2, par_sre_frf1, par_sre_frf2, par_sre_brf1, par_sre_brf2, par_sre_rg1, par_sre_rg2, par_output_ahist
} swift_parameter_id;
struct {
	 swift_parameter_id misprob, runs, nsims, delta0, delta1, asym, eta, alpha, beta, gamma, minact, theta, msac0, msac, h, h1, ppf, iota, refix, misfac, kappa0, kappa1, proc, decay, tau_l, tau_n, tau_ex, aord, cord, lord, nord, xord, ocshift, omn_fs1, omn_fs2, omn_sk1, omn_sk2, omn_frf1, omn_frf2, omn_brf1, omn_brf2, omn_rg1, omn_rg2, sre_fs1, sre_fs2, sre_sk1, sre_sk2, sre_frf1, sre_frf2, sre_brf1, sre_brf2, sre_rg1, sre_rg2, output_ahist;
} swift_parameter_ids = {par_misprob, par_runs, par_nsims, par_delta0, par_delta1, par_asym, par_eta, par_alpha, par_beta, par_gamma, par_minact, par_theta, par_msac0, par_msac, par_h, par_h1, par_ppf, par_iota, par_refix, par_misfac, par_kappa0, par_kappa1, par_proc, par_decay, par_tau_l, par_tau_n, par_tau_ex, par_aord, par_cord, par_lord, par_nord, par_xord, par_ocshift, par_omn_fs1, par_omn_fs2, par_omn_sk1, par_omn_sk2, par_omn_frf1, par_omn_frf2, par_omn_brf1, par_omn_brf2, par_omn_rg1, par_omn_rg2, par_sre_fs1, par_sre_fs2, par_sre_sk1, par_sre_sk2, par_sre_frf1, par_sre_frf2, par_sre_brf1, par_sre_brf2, par_sre_rg1, par_sre_rg2, par_output_ahist};
swift_parameter_meta swift_parameters_meta[] = {
	{"misprob", PARTYPE_DOUBLE, par_misprob},
	{"runs", PARTYPE_INTEGER, par_runs},
	{"nsims", PARTYPE_INTEGER, par_nsims},
	{"delta0", PARTYPE_DOUBLE, par_delta0},
	{"delta1", PARTYPE_DOUBLE, par_delta1},
	{"asym", PARTYPE_DOUBLE, par_asym},
	{"eta", PARTYPE_DOUBLE, par_eta},
	{"alpha", PARTYPE_DOUBLE, par_alpha},
	{"beta", PARTYPE_DOUBLE, par_beta},
	{"gamma", PARTYPE_DOUBLE, par_gamma},
	{"minact", PARTYPE_DOUBLE, par_minact},
	{"theta", PARTYPE_DOUBLE, par_theta},
	{"msac0", PARTYPE_DOUBLE, par_msac0},
	{"msac", PARTYPE_DOUBLE, par_msac},
	{"h", PARTYPE_DOUBLE, par_h},
	{"h1", PARTYPE_DOUBLE, par_h1},
	{"ppf", PARTYPE_DOUBLE, par_ppf},
	{"iota", PARTYPE_DOUBLE, par_iota},
	{"refix", PARTYPE_DOUBLE, par_refix},
	{"misfac", PARTYPE_DOUBLE, par_misfac},
	{"kappa0", PARTYPE_DOUBLE, par_kappa0},
	{"kappa1", PARTYPE_DOUBLE, par_kappa1},
	{"proc", PARTYPE_DOUBLE, par_proc},
	{"decay", PARTYPE_DOUBLE, par_decay},
	{"tau_l", PARTYPE_DOUBLE, par_tau_l},
	{"tau_n", PARTYPE_DOUBLE, par_tau_n},
	{"tau_ex", PARTYPE_DOUBLE, par_tau_ex},
	{"aord", PARTYPE_DOUBLE, par_aord},
	{"cord", PARTYPE_DOUBLE, par_cord},
	{"lord", PARTYPE_DOUBLE, par_lord},
	{"nord", PARTYPE_DOUBLE, par_nord},
	{"xord", PARTYPE_DOUBLE, par_xord},
	{"ocshift", PARTYPE_DOUBLE, par_ocshift},
	{"omn_fs1", PARTYPE_DOUBLE, par_omn_fs1},
	{"omn_fs2", PARTYPE_DOUBLE, par_omn_fs2},
	{"omn_sk1", PARTYPE_DOUBLE, par_omn_sk1},
	{"omn_sk2", PARTYPE_DOUBLE, par_omn_sk2},
	{"omn_frf1", PARTYPE_DOUBLE, par_omn_frf1},
	{"omn_frf2", PARTYPE_DOUBLE, par_omn_frf2},
	{"omn_brf1", PARTYPE_DOUBLE, par_omn_brf1},
	{"omn_brf2", PARTYPE_DOUBLE, par_omn_brf2},
	{"omn_rg1", PARTYPE_DOUBLE, par_omn_rg1},
	{"omn_rg2", PARTYPE_DOUBLE, par_omn_rg2},
	{"sre_fs1", PARTYPE_DOUBLE, par_sre_fs1},
	{"sre_fs2", PARTYPE_DOUBLE, par_sre_fs2},
	{"sre_sk1", PARTYPE_DOUBLE, par_sre_sk1},
	{"sre_sk2", PARTYPE_DOUBLE, par_sre_sk2},
	{"sre_frf1", PARTYPE_DOUBLE, par_sre_frf1},
	{"sre_frf2", PARTYPE_DOUBLE, par_sre_frf2},
	{"sre_brf1", PARTYPE_DOUBLE, par_sre_brf1},
	{"sre_brf2", PARTYPE_DOUBLE, par_sre_brf2},
	{"sre_rg1", PARTYPE_DOUBLE, par_sre_rg1},
	{"sre_rg2", PARTYPE_DOUBLE, par_sre_rg2},
	{"output_ahist", PARTYPE_INTEGER, par_output_ahist},
	{0, 0, 0}
};
void* swift_param_addr(swift_parameters* pars, swift_parameter_id id) {
	switch(id) {
		case par_misprob: return &pars->misprob;
		case par_runs: return &pars->runs;
		case par_nsims: return &pars->nsims;
		case par_delta0: return &pars->delta0;
		case par_delta1: return &pars->delta1;
		case par_asym: return &pars->asym;
		case par_eta: return &pars->eta;
		case par_alpha: return &pars->alpha;
		case par_beta: return &pars->beta;
		case par_gamma: return &pars->gamma;
		case par_minact: return &pars->minact;
		case par_theta: return &pars->theta;
		case par_msac0: return &pars->msac0;
		case par_msac: return &pars->msac;
		case par_h: return &pars->h;
		case par_h1: return &pars->h1;
		case par_ppf: return &pars->ppf;
		case par_iota: return &pars->iota;
		case par_refix: return &pars->refix;
		case par_misfac: return &pars->misfac;
		case par_kappa0: return &pars->kappa0;
		case par_kappa1: return &pars->kappa1;
		case par_proc: return &pars->proc;
		case par_decay: return &pars->decay;
		case par_tau_l: return &pars->tau_l;
		case par_tau_n: return &pars->tau_n;
		case par_tau_ex: return &pars->tau_ex;
		case par_aord: return &pars->aord;
		case par_cord: return &pars->cord;
		case par_lord: return &pars->lord;
		case par_nord: return &pars->nord;
		case par_xord: return &pars->xord;
		case par_ocshift: return &pars->ocshift;
		case par_omn_fs1: return &pars->omn_fs1;
		case par_omn_fs2: return &pars->omn_fs2;
		case par_omn_sk1: return &pars->omn_sk1;
		case par_omn_sk2: return &pars->omn_sk2;
		case par_omn_frf1: return &pars->omn_frf1;
		case par_omn_frf2: return &pars->omn_frf2;
		case par_omn_brf1: return &pars->omn_brf1;
		case par_omn_brf2: return &pars->omn_brf2;
		case par_omn_rg1: return &pars->omn_rg1;
		case par_omn_rg2: return &pars->omn_rg2;
		case par_sre_fs1: return &pars->sre_fs1;
		case par_sre_fs2: return &pars->sre_fs2;
		case par_sre_sk1: return &pars->sre_sk1;
		case par_sre_sk2: return &pars->sre_sk2;
		case par_sre_frf1: return &pars->sre_frf1;
		case par_sre_frf2: return &pars->sre_frf2;
		case par_sre_brf1: return &pars->sre_brf1;
		case par_sre_brf2: return &pars->sre_brf2;
		case par_sre_rg1: return &pars->sre_rg1;
		case par_sre_rg2: return &pars->sre_rg2;
		case par_output_ahist: return &pars->output_ahist;
	}
	stop(1, "Unknown parameter id!");
	return NULL;
}
