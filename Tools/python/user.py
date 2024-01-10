import os


if os.environ["USER"] in ["dennis.schwarz"]:
    postprocessing_output_directory = "/scratch-cbe/users/dennis.schwarz/MLunfolding/nanoTuples"
    postprocessing_tmp_directory    = "/scratch/hephy/cms/dennis.schwarz/MLunfolding/tmp/"
    plot_directory                  = "/groups/hephy/cms/dennis.schwarz/www/MLunfolding/plots"
    cache_dir                       = "/groups/hephy/cms/dennis.schwarz/MLunfolding/caches"
    analysis_results                = "/groups/hephy/cms/dennis.schwarz/MLunfolding/results/v1"
    cern_proxy_certificate          = "/users/dennis.schwarz/.private/.proxy"
