clear; clc; close all;
%Global parameters
A   = 1;            % cosine amplitude
f0  = 1e3;          % frequency in Hz
w0  = 2*pi*f0;      % angular frequency
T   = 1/f0;         % period

Mlist = [3 5 8 16 65]; % truncation orders to demonstrate
Mmax  = max(Mlist); % maximum order for coefficient computation

Nvec  = 2.^(5:10);  % number of sub-intervals for rectangle rule
idxRect = numel(Nvec);  % pick finest h = T/Nvec(idxRect)

%Walsh basis preprocessing (Hadamard) 
Hsize = 2^nextpow2(Mmax);
H     = hadamard(Hsize);
trans = sum(abs(diff(H,1,2)),2)/2;
[~, seqOrder] = sort(trans);
H = H(seqOrder,:);

%Part 1 : Compute coefficients c_n
x_fun = @(t) signal_xt(t,A,w0,T);
walsh = @(n,t) walsh_fun(n,t,T,H);

CN_quad = zeros(Mmax,1);
CN_rect_1 = zeros(Mmax,1);
CN_rect_2 = zeros(Mmax,1);

for n = 0:Mmax-1
    CN_quad(n+1) = (1/T) * integral(@(tt)x_fun(tt).*walsh(n,tt), ...
                                   0,T,'ArrayValued',true);
end

h_1 = T/Nvec(idxRect); %the most accurate h in rectangular method
ti_1 = 0:h_1:T-h_1;
x_rect_1 = x_fun(ti_1);

for n = 0:Mmax-1
    phi = walsh(n,ti_1);
    CN_rect_1(n+1) = h_1 * sum(x_rect_1 .* phi) / T;
end

h_2 = T/Nvec(1); %the least accurate h in rectangular method
ti_2 = 0:h_2:T-h_2;
x_rect_2 = x_fun(ti_2);

for n = 0:Mmax-1
    phi = walsh(n,ti_2);
    CN_rect_2(n+1) = h_2 * sum(x_rect_2 .* phi) / T;
end

%effect of step h on coeff error
h_vec   = T./Nvec;                     % rectangle widths for all Nvec
K       = numel(Nvec);

%compute rectangle-rule coefficients for *each* h (keep local variable)
CN_rect_all = zeros(Mmax,K);
for k = 1:K
    h_k  = h_vec(k);
    ti   = 0:h_k:T-h_k;
    x_rk = x_fun(ti);
    for n = 0:Mmax-1
        phi                = walsh(n,ti);
        CN_rect_all(n+1,k) = h_k*sum(x_rk.*phi)/T;
    end
end

errMat = abs(CN_rect_all - CN_quad);   % |c_n^{rect}(h) – c_n^{quad}|
errMat(errMat==0) = eps;               % avoid log(0)

%sort h_vec and errMat before plotting
[h_vec_sorted, idx_sort] = sort(h_vec, 'descend');  % h large to small
errMat_sorted = errMat(:, idx_sort);

figure('Name','Rectangle-rule coefficient error vs h', ...
       'NumberTitle','off'); hold on

showIdx = [3 7 11 15];                % n values to display
for k = 1:numel(showIdx)
    n = showIdx(k);
    semilogy(h_vec, errMat(n+1,:),'-o','LineWidth',1.2, ...
             'DisplayName',sprintf('n = %d',n));
end

set(gca,'XGrid','on','YGrid','on');
xlabel('h  (rectangle width)');
ylabel('|c_n^{rect} - c_n^{quad}|');
title('Influence of step size h on the accuracy of coefficients');
legend('Location','eastoutside');
drawnow;


%Part 2 : Plot reconstructions for different M
Nt_plot = 4000;
t_dense = linspace(0,T,Nt_plot);
x_true  = x_fun(t_dense);

WalshMat = zeros(Mmax,Nt_plot);
for n = 0:Mmax-1
    WalshMat(n+1,:) = walsh(n,t_dense);
end

for Mm = Mlist
    %First figure: accurate reconstruction (integral + fine h)
    figure('Name',sprintf('Reconstruction (accurate) M=%d',Mm), 'NumberTitle','off');
    plot(t_dense*1e3, x_true,'k-','LineWidth',1.5); hold on;
    plot(t_dense*1e3, CN_quad(1:Mm).'*WalshMat(1:Mm,:), 'r--','LineWidth',1.2);
    plot(t_dense*1e3, CN_rect_1(1:Mm).'*WalshMat(1:Mm,:), 'b:','LineWidth',1.2);
    xlabel('t [ms]'); ylabel('Amplitude');
    title(sprintf('Fourier–Walsh reconstruction by two methods (accurate h), M = %d',Mm));
    legend('True','integral','rectangle fine h','Location','Best');
    grid on;
    
    %Second figure: inaccurate reconstruction (integral + coarse h)
    figure('Name',sprintf('Reconstruction (coarse) M=%d',Mm), 'NumberTitle','off');
    plot(t_dense*1e3, x_true,'k-','LineWidth',1.5); hold on;
    plot(t_dense*1e3, CN_quad(1:Mm).'*WalshMat(1:Mm,:), 'r--','LineWidth',1.2);
    plot(t_dense*1e3, CN_rect_2(1:Mm).'*WalshMat(1:Mm,:), 'g-.','LineWidth',1.2);
    xlabel('t [ms]'); ylabel('Amplitude');
    title(sprintf('Fourier–Walsh reconstruction by two methods (inaccurate h), M = %d',Mm));
    legend('True','integral','rectangle coarse h','Location','Best');
    grid on;
end


%Part 3 : Compute & plot mean-square error ε_M
Mvec = Mlist;               % 2, 4, 8, 16   (no intermediate points)

E0   = A^2 * T / 4;         % analytical ∫|x|^2 dt
eps_quad = zeros(size(Mvec));
eps_rect_1 = zeros(size(Mvec));
eps_rect_2 = zeros(size(Mvec));

for k = 1:numel(Mvec)
    m = Mvec(k);
    eps_quad(k) = E0 - T*sum(abs(CN_quad(1:m)).^2);
    eps_rect_1(k) = E0 - T*sum(abs(CN_rect_1(1:m)).^2);
    eps_rect_2(k) = E0 - T*sum(abs(CN_rect_2(1:m)).^2);
end

%Accurate methods (integral + rectangle fine h)
figure('Name','Mean-square error vs M (accurate)','NumberTitle','off');
loglog(Mvec, eps_quad,'ro-','LineWidth',1.5, 'DisplayName','integral()'); hold on; grid on;
loglog(Mvec, eps_rect_1,'bs-','LineWidth',1.5, ...
    'DisplayName',sprintf('rectangle fine h (h = T/%d)', Nvec(idxRect)));
xlabel('M  (number of Walsh terms)');
ylabel('\epsilon_M  (mean-square error)');
title('Approximation error of Fourier–Walsh truncation in two methods (accurate h)');
legend('Location','southwest');

%Inaccurate method (rectangle coarse h)
figure('Name','Mean-square error vs M (coarse h)','NumberTitle','off');
loglog(Mvec, eps_quad,'ro-','LineWidth',1.5, 'DisplayName','integral()'); hold on; grid on;
loglog(Mvec, eps_rect_2,'g-.','LineWidth',1.5, ...
    'DisplayName',sprintf('rectangle coarse h (h = T/%d)', Nvec(1)));
xlabel('M  (number of Walsh terms)');
ylabel('\epsilon_M  (mean-square error)');
title('Approximation error of Fourier–Walsh truncation in two methods (inaccurate h)');
legend('Location','southwest');

%Local functions
function x = signal_xt(t,A,w0,T)
    tau = mod(t+T/2,T)-T/2;
    x   = A*cos(w0*tau).*(abs(tau)<=T/4);
end

function phi = walsh_fun(n,t,T,H)
    idx = floor(mod(t,T)/T*size(H,1)) + 1;
    phi = H(n+1,idx);
end
