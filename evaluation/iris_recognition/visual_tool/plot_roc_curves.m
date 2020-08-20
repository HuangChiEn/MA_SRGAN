function plot_roc_curves(load_cell, legnd_str_lst, plot_params)
    fig = figure;
    for idx=1:length(load_cell)
        load(load_cell{idx});
        plotRocCurve(results, plot_params{idx}); 
        hold on;
    end
    legend(legnd_str_lst);
    saveas(fig, './result_fig/ROC_fig.bmp');
end