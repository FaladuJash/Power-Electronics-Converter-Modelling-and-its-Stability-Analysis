%% Frequency domain analysis
% Author: Francisco Javier Cifuentes Garcia
% Date: September 2024
% 1) Grid impedance and VSC admittance computation
% 2) Application of the Generalized Nyquist Criterion (GNC)
% 3) Oscillation mode identification via eigenvalue decomposition
% More info on the method: Cifuentes Garcia, F.J. et al. "Automated Frequency-Domain
% Small-Signal Stability Analysis of Electrical Energy Hubs", 2024 IEEE PES ISGT-Europe.
% The input of this script is a state-space model representing the dq-frame admittance
% of a VSC at its point of connection.

%% Define the grid impedance
% Please, note that modification of the grid-side impedance might also change
% the VSC operating point and thus its linearized model (admittance) so it is
% recommended to update the steady-state op. point based on the actual tested
% impedance before obtaining the linearized model.

s = tf('s'); % Complex frequency
l_g = 1/2; % Reactance in per unit
r_g = l_g/10; % Resistance in per unit
c_g = 1/(0.4*l_g); % Series capacitance in pu

% Series RL
% z_th = r_g*eye(2) + l_g*s/wn*eye(2) + l_g*Wpu;

% Series RLC
z_th = r_g*eye(2) + l_g*s/wn*eye(2) + l_g*Wpu + 1/c_g*inv(s/wn*eye(2) + Wpu); 

%% Frequency response: VSC-side and grid-side
fmax = 1000;fmin = 1e-2; % [Hz] Define the freq. range of interest
points = 2000; % Number of frequency points
omegas = 2*pi*logspace(log10(fmin),log10(fmax),points); freq = omegas/2/pi; % [rad/s] and [Hz]

% Grid equivalent
[mag_dd,ph_dd] = bode(z_th(1,1),omegas); Z_dd = squeeze(mag_dd).*exp(1i*squeeze(ph_dd)*pi/180);
[mag_dq,ph_dq] = bode(z_th(1,2),omegas); Z_dq = squeeze(mag_dq).*exp(1i*squeeze(ph_dq)*pi/180);
[mag_qd,ph_qd] = bode(z_th(2,1),omegas); Z_qd = squeeze(mag_qd).*exp(1i*squeeze(ph_qd)*pi/180);
[mag_qq,ph_qq] = bode(z_th(2,2),omegas); Z_qq = squeeze(mag_qq).*exp(1i*squeeze(ph_qq)*pi/180);

% VSC: "model" is a SS model with v_dq as input and i_dq as output, where the current is
% positive into the converter
[mag_dd,ph_dd] = bode(model(1,1),omegas); Y_dd = squeeze(mag_dd).*exp(1i*squeeze(ph_dd)*pi/180);
[mag_dq,ph_dq] = bode(model(1,2),omegas); Y_dq = squeeze(mag_dq).*exp(1i*squeeze(ph_dq)*pi/180);
[mag_qd,ph_qd] = bode(model(2,1),omegas); Y_qd = squeeze(mag_qd).*exp(1i*squeeze(ph_qd)*pi/180);
[mag_qq,ph_qq] = bode(model(2,2),omegas); Y_qq = squeeze(mag_qq).*exp(1i*squeeze(ph_qq)*pi/180);

% Arrange into multi-dim arrays with size 2 x 2 x # points
Z = zeros(2,2,length(omegas)); % Initialize grid impedance
Y = zeros(2,2,length(omegas)); % Initialize VSC admittance
for f = 1:points
    Z(:,:,f) = [Z_dd(f),Z_dq(f);Z_qd(f),Z_qq(f)]; % Save the grid impedance for every frequency
    Y(:,:,f) = [Y_dd(f),Y_dq(f);Y_qd(f),Y_qq(f)]; % Save the VSC admittance for every frequency
end
save('freq_response.mat','Z','Y','freq','points'); % Optional: save the data

%% Generalized Nyquist Criterion (GNC)
lambdas = zeros(2,points); % Array of eigenvalues over the frequency
tic
for k = 1:points
    Ek = eig(Z(:,:,k)*Y(:,:,k));
    if k > 1
        %         It compares the current vector of eigenvalues to the vector of
        %         eigenvalues at the previous frequency. If the flipped current one
        %         is closer to the prior than the unflipped, use the flipped.
        %         Otherwise, use the unflipped. A more general approach is to use
        %         eigenshuffle()
        if norm(Ek-lambdas(:,k-1), 'Inf') > norm(flip(Ek)-lambdas(:,k-1), 'Inf')
            Ek = flip(Ek);
        end
    end
    lambdas(:,k) = Ek;
end
toc

% Nyquist plot
figure('Name','Nyquist')
plot(real(lambdas(2,:)),imag(lambdas(2,:)),'m','LineWidth',2);hold on
plot(real(lambdas(1,:)),imag(lambdas(1,:)),'b','LineWidth',2)
plot(-1,0,'kx','MarkerSize',10,'LineWidth',1);grid on
plot(cos(-pi:1e-3:pi),sin(-pi:1e-3:pi),'k--','Linewidth',1)
plot(real(lambdas(2,:)),-imag(lambdas(2,:)),'m','LineWidth',2);hold on
plot(real(lambdas(1,:)),-imag(lambdas(1,:)),'b','LineWidth',2)
sgtitle(strcat("Nyquist Plot: VSC + grid equivalent with $|Z_g| =$ ",num2str(sqrt(r_g^2+(l_g-1/c_g)^2),'%.2f')," pu"))
xlabel("$\Re\{\lambda\}$]");ylabel("$\Im\{\lambda\}$")
legend('$\lambda_\mathrm{1}$','$\lambda_\mathrm{2}$','$(-1,0j)$','interpreter','latex','NumColumns',1)

axes('position',[.50 .2 .35 .3]);box on
plot(real(lambdas(2,:)),imag(lambdas(2,:)),'m');hold on;grid on
plot(real(lambdas(2,:)),-imag(lambdas(2,:)),'m')
plot(real(lambdas(1,:)),imag(lambdas(1,:)),'b')
plot(real(lambdas(1,:)),-imag(lambdas(1,:)),'b')
plot(-1,0,'kx')
plot(cos(-pi:1e-3:pi),sin(-pi:1e-3:pi),'k--','Linewidth',1)
xlim([-2 2]);ylim([-2 2]);xticks(-2:1:2);yticks(-2:1:2)
title("Detail around the unit circle")
set(gcf,'position',[50,50,800,720])

%% Oscillation mode identification via EVD
% Very low damped modes appear as high-magnitude eigenvalue peaks
evd = zeros(2,points); % Array of closed-loop eigenvalues over the frequency
for k = 1:points
    Ek = eig(inv(Z(:,:,k)) + Y(:,:,k));
    if k > 1
        % It compares the current vector of eigenvalues to the vector of
        % eigenvalues at the previous frequency. If the flipped current one
        % is closer to the prior than the unflipped, use the flipped.
        % Otherwise, use the unflipped. A more general approach is to use
        % eigenshuffle()
        if norm(Ek-evd(:,k-1), 'Inf') > norm(flip(Ek)-evd(:,k-1), 'Inf')
            Ek = flip(Ek);
        end
    end
    evd(:,k) = Ek;
end
toc

% Plot the EVD
figure('Name','EVD')
hold on; box on; grid on
title('Oscillation mode identification','interpreter','latex');
set(gca,'xScale', 'log');set(gca,'yScale', 'log');
xlabel("Frequency [Hz]");ylabel("$$|\lambda_i(\mathbf{Z}_{Closed-loop})|$$ [pu]")
set(gcf,'Units','centimeters','position',[15,5,25,10])
plot(freq,1./abs(evd(1,:)),"b-","LineWidth",2);hold on
plot(freq,1./abs(evd(2,:)),"r-","LineWidth",2);hold on
xlim([min(freq),max(freq)]);
legend('$|\lambda_\mathrm{1}|$','$|\lambda_\mathrm{2}|$','interpreter','latex','NumColumns',1)
grid off; grid
