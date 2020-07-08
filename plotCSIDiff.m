function plotCSIDiff(Y,Title,Trace,color,snr,n)
    load('EAngle.mat');
    EAngle = cell2mat(EAngle(n));
    figure('Name',strcat('CSI Diagram LEO_Track_',num2str(Trace),'_SNR_',num2str(snr)));
    realDiff = real(Y(2:end)) - real(Y(1:end-1)); 
    imagDiff = imag(Y(2:end)) - imag(Y(1:end-1)); 
    plot(1:size(realDiff),realDiff,'color',color(1),'LineWidth',1); hold on;
    plot(1:size(imagDiff),imagDiff,'color',color(2),'LineWidth',1); hold off;
    % set(gca,'xtick',1:floor(size(Y)/8):size(Y),'xticklabel',round(EAngle(1:floor(size(Y)/8):size(Y))));
    
    title('CSI Diff','Interpreter','latex');
    
    xlabel('Elevation Angle (deg)','Interpreter','latex');
    ylabel('CSI Value $(h)$','Interpreter','latex');
    legend('Real $(h)$','Imag $(h)$','Interpreter','latex','Location','southwest');
    % legend('Imag $(h)$','Interpreter','latex','Location','southwest');
    legend('boxoff');
    
end