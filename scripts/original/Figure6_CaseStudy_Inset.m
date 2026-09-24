clear; close all; clc;
st = p05_style();
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root,'chemistry','Figure6_case_studies.csv'), ...
              'VariableNamingRule','preserve');

names = {'Acetaminophen oxidation', ...
         'Orphenadrine dealkylation', ...
         'Mercaptopurine S-methylation'};

Y = [T.delta_electrophilicity_eV, T.delta_dft_electrophilicity_eV];

fig = figure('Units','centimeters','Position',[2 2 11.5 6.7], ...
             'Color','w','Name','Figure 6 quantitative inset');
ax = axes(fig); hold(ax,'on');

b = barh(ax,1:height(T),Y,'grouped','EdgeColor','none');
b(1).FaceColor = st.darkGray;
b(2).FaceColor = st.midGray;

yticks(ax,1:height(T));
yticklabels(ax,names);
set(ax,'YDir','reverse');
xlabel(ax,'\Delta\omega (eV)','Interpreter','tex', ...
    'FontName',st.font,'FontSize',st.labelSize);
xline(ax,0,'-','Color',st.lightGray,'LineWidth',0.8);
legend(ax,{'GFN2-xTB','\omegaB97X-D'}, ...
    'Interpreter','tex','Location','southoutside', ...
    'Orientation','horizontal','Box','off', ...
    'FontName',st.font,'FontSize',st.fontSize);

% Generous padding prevents the large acetaminophen bar from contacting
% the right frame, while the negative mercaptopurine bar remains visible.
lo = min(Y,[],'all'); hi = max(Y,[],'all');
span = hi-lo;
xlim(ax,[lo-0.10*span hi+0.14*span]);
grid(ax,'on'); ax.YGrid='off';
finish_p05_axes(ax);

export_p05(fig,'Figure6_case_study_quantum_inset');
