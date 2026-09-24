clear; close all; clc;
st = p05_style();
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root,'data','final_external_metrics.csv'), ...
              'VariableNamingRule','preserve');

fig = figure('Units','centimeters','Position',[2 2 18 13.2], ...
             'Color','w','Name','Figure 3 - External validation');
tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

letters = {'A','B','C','D'};

for d = 1:numel(st.datasetOrder)
    ax = nexttile(tl,d); hold(ax,'on');
    S = T(strcmp(T.dataset,st.datasetOrder{d}),:);

    nThis = S.n(1);
    posThis = S.positives(1);
    negThis = S.negatives(1);

    % Chance line
    xline(ax,0.5,'--','Color',st.midGray,'LineWidth',0.9);

    for i = 1:numel(st.modelOrder)
        row = S(strcmp(S.model,st.modelOrder{i}),:);
        x = row.auroc;
        y = 5-i; % ECFP at top
        plot(ax,[0.5 x],[y y],'-','Color',st.lightGray,'LineWidth',1.0);
        scatter(ax,x,y,58, ...
            'MarkerFaceColor',st.modelColors(i,:), ...
            'MarkerEdgeColor','w','LineWidth',0.8);
    end

    yticks(ax,1:4);
    yticklabels(ax,fliplr(st.modelLabels));
    ylim(ax,[0.45 4.55]);
    xlim(ax,[0.48 1.015]);
    xticks(ax,0.5:0.1:1.0);
    xlabel(ax,'AUROC','FontName',st.font,'FontSize',st.labelSize);
    title(ax,sprintf('%s  (n=%d; %d+/%d-)', ...
        st.datasetLabels{d},nThis,posThis,negThis), ...
        'FontName',st.font,'FontSize',st.labelSize,'FontWeight','normal');

    grid(ax,'on');
    ax.YGrid = 'off';
    finish_p05_axes(ax);
    panel_letter(ax,letters{d});
end

% Exact values are intentionally omitted from the plotting area to prevent
% collisions in the very tight DILImap panels.
export_p05(fig,'Figure3_external_validation');
