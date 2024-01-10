import os


if os.environ["USER"] in ["dennis.schwarz"]:
    postprocessing_output_directory = "/scratch-cbe/users/dennis.schwarz/MLUnfolding/nanoTuples"
    postprocessing_tmp_directory    = "/scratch/hephy/cms/dennis.schwarz/MLUnfolding/tmp/"
    plot_directory                  = "/groups/hephy/cms/dennis.schwarz/www/MLUnfolding/plots"
    cache_dir                       = "/groups/hephy/cms/dennis.schwarz/MLUnfolding/caches"
    analysis_results                = "/groups/hephy/cms/dennis.schwarz/MLUnfolding/results/v1"
    cern_proxy_certificate          = "/users/dennis.schwarz/.private/.proxy"
