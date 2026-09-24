clear; close all; clc;
st = p05_style();
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root,'data','final_dili_fold_metrics.csv'), ...
              'VariableNamingRule','preserve');

fig = figure('Units','centimeters','Position',[2 2 18 13.5], ...
             'Color','w','Name','Figure 2 - DILI benchmark');
tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

metrics = {'auroc','auprc','mcc','brier'};
ylabs   = {'AUROC','AUPRC','Matthews correlation coefficient','Brier score'};
letters = {'A','B','C','D'};

rng(11); % deterministic jitter

for p = 1:4
    ax = nexttile(tl,p); hold(ax,'on');

    allY = [];
    for i = 1:numel(st.modelOrder)
        y = T.(metrics{p})(strcmp(T.model,st.modelOrder{i}));
        y = y(isfinite(y));
        allY = [allY; y]; %#ok<AGROW>

        % Box first; raw fold points are deliberately faint to prevent clutter.
        boxchart(ax, repmat(i,numel(y),1), y, ...
            'BoxFaceColor',st.modelColors(i,:), ...
            'BoxFaceAlpha',0.44, ...
            'WhiskerLineColor',st.midGray, ...
            'MarkerStyle','none', ...
            'BoxWidth',0.52);

        xj = i + 0.16*(rand(size(y))-0.5);
        scatter(ax,xj,y,11, ...
            'MarkerFaceColor',st.modelColors(i,:), ...
            'MarkerEdgeColor','none', ...
            'MarkerFaceAlpha',0.20);

        plot(ax,i,mean(y,'omitnan'),'d', ...
            'MarkerSize',5.5,'MarkerFaceColor','w', ...
            'MarkerEdgeColor',st.darkGray,'LineWidth',0.9);
    end

    xlim(ax,[0.45 4.55]);
    xticks(ax,1:4);
    xticklabels(ax,st.modelLabels);
    xtickangle(ax,14);
    ylabel(ax,ylabs{p},'FontName',st.font,'FontSize',st.labelSize);

    lo = min(allY); hi = max(allY);
    span = max(hi-lo,0.05);
    pad = 0.12*span;

    if p <= 2
        ylim(ax,[max(0.45,lo-pad), min(1.00,hi+pad)]);
    elseif p == 3
        ylim(ax,[lo-pad, hi+pad]);
    else
        ylim(ax,[max(0,lo-pad), hi+pad]);
    end

    grid(ax,'on');
    ax.XGrid = 'off';
    finish_p05_axes(ax);
    panel_letter(ax,letters{p});
end

% No numerical labels are placed on top of points/boxes. Exact values belong
% in the manuscript table/caption, which keeps the figure uncluttered.
export_p05(fig,'Figure2_DILI_scaffold_CV');
