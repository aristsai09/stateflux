clear; close all; clc;
st = p05_style();
root = fileparts(fileparts(mfilename('fullpath')));

W  = readtable(fullfile(root,'data','dft_delta_omega.csv'), ...
               'VariableNamingRule','preserve');
IP = readtable(fullfile(root,'data','dft_delta_IP.csv'), ...
               'VariableNamingRule','preserve');
EA = readtable(fullfile(root,'data','dft_delta_EA.csv'), ...
               'VariableNamingRule','preserve');
C  = readtable(fullfile(root,'data','dft_validation_by_class.csv'), ...
               'VariableNamingRule','preserve');

fig = figure('Units','centimeters','Position',[2 2 18 14.0], ...
             'Color','w','Name','Figure 5 - DFT validation');
tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

classes = {'dealkylation','oxidation','phaseII_conjugation'};
classLabels = {'Dealkylation','Oxidation','Phase II'};
markers = {'o','s','^'};

% A: delta omega
ax1 = nexttile(tl,1);
scatter_panel(ax1,W,'delta_electrophilicity_eV', ...
    'delta_dft_electrophilicity_eV', ...
    '\Delta\omega_{xTB} (eV)','\Delta\omega_{DFT} (eV)', ...
    classes,classLabels,markers,st,true);
panel_letter(ax1,'A');

% Label only the mechanistically central sentinel. No other point labels are
% drawn, which prevents dense text collisions.
idx = strcmp(W.edge_id,'R05E014');
if any(idx)
    smart_label(ax1,W.delta_electrophilicity_eV(idx), ...
        W.delta_dft_electrophilicity_eV(idx),'Acetaminophen',st);
end

% B: delta IP
ax2 = nexttile(tl,2);
scatter_panel(ax2,IP,'delta_vip_eV','delta_dft_ip_eV', ...
    '\DeltaIP_{xTB} (eV)','\DeltaIP_{DFT} (eV)', ...
    classes,classLabels,markers,st,false);
panel_letter(ax2,'B');

% C: delta EA
ax3 = nexttile(tl,3);
scatter_panel(ax3,EA,'delta_vea_eV','delta_dft_ea_eV', ...
    '\DeltaEA_{xTB} (eV)','\DeltaEA_{DFT} (eV)', ...
    classes,classLabels,markers,st,false);
panel_letter(ax3,'C');

% D: transformation-class error
ax4 = nexttile(tl,4); hold(ax4,'on');
[~,loc] = ismember(classes,C.broad_class);
C = C(loc,:);
Y = [C.mae_delta_omega_eV C.rmse_delta_omega_eV];
b = bar(ax4,1:3,Y,'grouped','EdgeColor','none');

% Deliberately neutral tones so panel D does not compete with the class
% colors used in the correlation panels.
b(1).FaceColor = st.darkGray;
b(2).FaceColor = st.midGray;

xticks(ax4,1:3);
xticklabels(ax4,{'Dealkylation (n=10)','Oxidation (n=6)','Phase II (n=8)'});
xtickangle(ax4,12);
ylabel(ax4,'Error in \Delta\omega (eV)','Interpreter','tex', ...
    'FontName',st.font,'FontSize',st.labelSize);
legend(ax4,{'MAE','RMSE'},'Location','northwest','Box','off', ...
    'FontName',st.font,'FontSize',st.fontSize);
ylim(ax4,[0 max(Y,[],'all')*1.22]);
grid(ax4,'on'); ax4.XGrid='off';
finish_p05_axes(ax4);
panel_letter(ax4,'D');

export_p05(fig,'Figure5_xTB_DFT_validation');

% -------------------------------------------------------------------------
function scatter_panel(ax,T,xvar,yvar,xlab,ylab,classes,classLabels,markers,st,showLegend)
hold(ax,'on');

xall = T.(xvar);
yall = T.(yvar);
good = isfinite(xall) & isfinite(yall);
xall = xall(good);
yall = yall(good);

lo = min([xall; yall]);
hi = max([xall; yall]);
span = max(hi-lo,0.25);
pad = 0.09*span;
lim = [lo-pad hi+pad];

plot(ax,lim,lim,'-','Color',st.lightGray,'LineWidth',1.0);

h = gobjects(numel(classes),1);
for k = 1:numel(classes)
    idx = strcmp(T.broad_class,classes{k});
    h(k)=scatter(ax,T.(xvar)(idx),T.(yvar)(idx),52, ...
        'Marker',markers{k}, ...
        'MarkerFaceColor',st.classColors(k,:), ...
        'MarkerEdgeColor','w','LineWidth',0.7);
end

% Regression line
p = polyfit(xall,yall,1);
xx = linspace(lim(1),lim(2),100);
plot(ax,xx,polyval(p,xx),'--','Color',st.darkGray,'LineWidth',1.05);

R = corrcoef(xall,yall);
r = R(1,2);
text(ax,0.04,0.94,sprintf('Pearson r = %.3f',r), ...
    'Units','normalized','HorizontalAlignment','left', ...
    'VerticalAlignment','top','FontName',st.font, ...
    'FontSize',st.fontSize,'Color',st.darkGray);

xlim(ax,lim); ylim(ax,lim);
axis(ax,'square');
xlabel(ax,xlab,'Interpreter','tex','FontName',st.font,'FontSize',st.labelSize);
ylabel(ax,ylab,'Interpreter','tex','FontName',st.font,'FontSize',st.labelSize);
grid(ax,'on');
finish_p05_axes(ax);

if showLegend
    legend(ax,h,classLabels,'Location','northwest','Box','off', ...
        'FontName',st.font,'FontSize',st.fontSize-0.5);
end
end

function smart_label(ax,x,y,label,st)
xl=xlim(ax); yl=ylim(ax);
dx=0.03*(xl(2)-xl(1));
dy=0.03*(yl(2)-yl(1));

% Acetaminophen sits near the upper-right corner, so place its label to
% the left/below rather than allowing it to run through the border.
if x > xl(1)+0.72*(xl(2)-xl(1))
    xtext=x-dx;
    ha='right';
else
    xtext=x+dx;
    ha='left';
end
if y > yl(1)+0.78*(yl(2)-yl(1))
    ytext=y-dy;
    va='top';
else
    ytext=y+dy;
    va='bottom';
end

text(ax,xtext,ytext,label, ...
    'HorizontalAlignment',ha,'VerticalAlignment',va, ...
    'FontName',st.font,'FontSize',st.fontSize-0.4, ...
    'Color',st.darkGray,'Clipping','on');
end
