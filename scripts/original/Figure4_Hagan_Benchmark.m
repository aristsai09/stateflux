clear; close all; clc;
st = p05_style();
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root,'data','hagan_metrics.csv'), ...
              'VariableNamingRule','preserve');

% Lock model order.
[~,loc] = ismember(st.modelOrder,T.model);
T = T(loc,:);

fig = figure('Units','centimeters','Position',[2 2 18 12.8], ...
             'Color','w','Name','Figure 4 - Hagan benchmark');
tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

metrics = {'auroc','auprc','mcc','brier'};
xlabels = {'AUROC','AUPRC','Matthews correlation coefficient','Brier score'};
letters = {'A','B','C','D'};

for p = 1:4
    ax = nexttile(tl,p); hold(ax,'on');

    vals = T.(metrics{p});
    bh = barh(ax,1:4,vals,0.60,'FaceColor','flat','EdgeColor','none');
    bh.CData = st.modelColors;

    yticks(ax,1:4);
    yticklabels(ax,st.modelLabels);
    set(ax,'YDir','reverse');

    % All four metrics lie on a 0-1 scale, so a shared honest zero baseline
    % makes cross-panel visual comparison straightforward.
    xlim(ax,[0 1]);
    xticks(ax,0:0.2:1.0);
    xlabel(ax,xlabels{p},'FontName',st.font,'FontSize',st.labelSize);

    grid(ax,'on');
    ax.YGrid = 'off';
    finish_p05_axes(ax);
    panel_letter(ax,letters{p});

    if p == 4
        text(ax,0.98,0.06,'lower is better','Units','normalized', ...
            'HorizontalAlignment','right','VerticalAlignment','bottom', ...
            'FontName',st.font,'FontSize',st.fontSize-0.5, ...
            'Color',st.midGray);
    end
end

export_p05(fig,'Figure4_Hagan_transfer_benchmark');
