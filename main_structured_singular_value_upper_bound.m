% Q-scaling upper bound of the structured singular value (input_output_analysis.pdf, Sec. 2)
%
%   mu_K[H_grad(kz,omega)] <= inf_{Q=Q^*>0, Q in C^{3x3}} sigma_max[ (Q (x) I_3N) H_grad (Q^{-1} (x) I_N) ]
%
% with H_grad = diag(grad,grad,grad) * H,  grad = [d/dx; d/dy; i*kz],  H = C (i*omega*E - A)^{-1} B.
%
% Ordering of H_grad (size 9N x 3N, N = Nx*Ny):
%   input  : [fx; fy; fz]                                  (3 blocks of N)
%   output : [ux;uy;uz | vx;vy;vz | wx;wy;wz]               (3 blocks of 3N)
% so Q (x) I_3N acts on the velocity-component blocks of the output and
% Q^{-1} (x) I_N on the forcing-component blocks of the input, exactly as
% k_xi = I_3 (x) F requires.
%
% Optimization:
%   sigma depends on Q only through P = Q^*Q (unitary invariance), and the
%   problem is invariant to scaling P -> c P.  We parametrize
%       P = L L^*,  L lower triangular,  L11 = 1,  L22 = exp(a2),  L33 = exp(a3),
%   (8 real unknowns), take Q = L^* (so Q^*Q = P), and minimize sigma with BFGS
%   using the analytic gradient
%       d(sigma^2) = tr(dP * Z^T),  Z_ij = y_i^* y_j - sigma^2 x_i^* x_j,
%   where x = Q_R^{-1} v, y = H_grad x, (sigma,u,v) is the top singular triplet
%   of the scaled operator.  Any Q gives a valid upper bound, so even an
%   early-stopped optimization returns a rigorous bound.
%
% Two evaluation modes:
%   'full'        : H assembled explicitly (resolvent_2D_xy), small grids only.
%   'matrix_free' : H applied via preconditioned GMRES (same as the main
%                   resolvent script), svds on a function handle.

clearvars; close all; clc

params.kz    = 1;
params.omega = 1;
params.Nx    = 16;
params.Ny    = 16;
params.N     = params.Nx*params.Ny;
params.Re    = 358;
params.Ly    = 2;
params.Lx    = 2*pi;
params.gmres_tol     = 1e-9;
params.gmres_restart = 250;
params.gmres_maxit   = params.Nx*params.Ny;
params.preconditioner_type = 'fft_1d_resolvent';   % or 'fft_1d_laplacian'

% 'full', 'matrix_free', or 'both' (both = run both and compare)
mode = 'both';

% 'auto'      : fminunc (Optimization Toolbox, analytic gradient) if installed,
%               otherwise the built-in BFGS below
% 'fminunc' | 'bfgs' | 'fminsearch' (derivative-free, base MATLAB)
opt.optimizer = 'auto';
opt.maxit     = 60;      % max iterations
opt.tol_f     = 1e-8;    % relative decrease tolerance
opt.tol_g     = 1e-7;    % gradient-norm tolerance
opt.max_step  = 1;       % max ||d theta|| per line search (theta ~ log-scale of Q)
opt.disp      = true;
opt.check_grad = true;   % finite-difference gradient check at the start

params = prepare_params(params);

results = struct();

if any(strcmp(mode,{'full','both'}))
    fprintf('\n===== full-matrix evaluation =====\n');
    t0 = cputime;
    Hgrad = build_H_grad_full(params);
    ops.apply = @(x,tflag) apply_full(Hgrad,x,tflag);
    ops.size  = size(Hgrad);
    ops.svds_tol = 1e-10;
    results.full = mu_upper_bound(ops,params,opt);
    results.full.cpu_s = cputime-t0;
end

if any(strcmp(mode,{'matrix_free','both'}))
    fprintf('\n===== matrix-free evaluation =====\n');
    t0 = cputime;
    params = add_preconditioner(params);
    ops.apply = @(x,tflag) apply_H_grad_mf(x,tflag,params);
    ops.size  = [9*params.N, 3*params.N];
    ops.svds_tol = 1e-8;
    opt.check_grad = false;
    results.matrix_free = mu_upper_bound(ops,params,opt);
    results.matrix_free.cpu_s = cputime-t0;
end

% ---------------- summary ----------------
names = fieldnames(results);
for k = 1:numel(names)
    r = results.(names{k});
    fprintf('\n[%s]  sigma_max(H_grad) (Q=I) = %.8e\n', names{k}, r.sigma_identity);
    fprintf('[%s]  mu upper bound (opt. Q)  = %.8e   (%d iters, %.1f s CPU)\n', ...
        names{k}, r.mu_upper, r.iterations, r.cpu_s);
    fprintf('[%s]  optimal P = Q^*Q (normalized, trace = 3):\n', names{k});
    disp(r.P);
end
if isfield(results,'full') && isfield(results,'matrix_free')
    fprintf('relative difference full vs matrix-free: %.3e\n', ...
        abs(results.full.mu_upper-results.matrix_free.mu_upper)/results.full.mu_upper);
end

for k = 1:numel(names)
    r = results.(names{k});
    figure('Name',names{k});
    semilogy(0:numel(r.history)-1, r.history, 'o-','LineWidth',1.3); hold on
    yline(r.sigma_identity,'k--','Q = I');
    xlabel(sprintf('iteration (%s)',r.optimizer)); ylabel('$\bar\sigma[Q_L H_\nabla Q_R^{-1}]$','Interpreter','latex');
    title(sprintf('Q-scaling upper bound (%s), k_z=%g, \\omega=%g',names{k},params.kz,params.omega),'Interpreter','tex');
    grid on
end


%% =====================================================================
%  Upper-bound optimization
%  =====================================================================
function res = mu_upper_bound(ops,params,opt)
    N = params.N;
    % handle object: keeps the svds warm start / history across calls made
    % by any optimizer (fminunc, fminsearch or the built-in BFGS)
    cache = containers.Map({'v0','hist'},{[],[]});
    fun = @(th) objective(th,ops,N,cache);

    theta0 = zeros(8,1);                   % theta = 0  <=>  Q = I
    res.sigma_identity = fun(theta0);
    if opt.check_grad, check_gradient(fun,0.1*sin(1:8).'); end

    method = opt.optimizer;
    if strcmp(method,'auto')
        if has_optim_toolbox(), method = 'fminunc'; else, method = 'bfgs'; end
    end
    fprintf('optimizer: %s\n',method);
    cache('hist') = res.sigma_identity;
    disp_mode = 'off'; if opt.disp, disp_mode = 'iter'; end
    outfun = @(x,ov,state) record_history(ov,state,cache);

    switch method
        case 'fminunc'
            o = optimoptions('fminunc','Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient',true,'MaxIterations',opt.maxit, ...
                'FunctionTolerance',opt.tol_f,'OptimalityTolerance',opt.tol_g, ...
                'StepTolerance',1e-10,'Display',disp_mode,'OutputFcn',outfun);
            [theta,f,~,out] = fminunc(fun,theta0,o);
            iters = out.iterations;
        case 'fminsearch'
            o = optimset('MaxIter',20*opt.maxit,'MaxFunEvals',40*opt.maxit, ...
                'TolFun',opt.tol_f*res.sigma_identity,'TolX',1e-6, ...
                'Display',disp_mode,'OutputFcn',outfun);
            [theta,f,~,out] = fminsearch(fun,theta0,o);
            iters = out.iterations;
        case 'bfgs'
            [theta,f,iters] = bfgs_armijo(fun,theta0,opt,cache);
        otherwise
            error('Unknown optimizer: %s',method);
    end

    [Q,~] = theta_to_Q(theta);
    P = Q'*Q;  P = 3*P/real(trace(P));
    res.mu_upper   = f;
    res.Q          = Q;
    res.P          = P;
    res.theta      = theta;
    res.history    = cache('hist');
    res.iterations = iters;
    res.optimizer  = method;
end


function tf = has_optim_toolbox()
    tf = ~isempty(ver('optim')) && license('test','Optimization_Toolbox');
end


function stop = record_history(ov,state,cache)
    stop = false;
    if strcmp(state,'iter') && ov.iteration > 0
        cache('hist') = [cache('hist'), ov.fval];
    end
end


function check_gradient(fun,th)
    h = 1e-6; gfd = zeros(numel(th),1);
    for k = 1:numel(th)
        e = zeros(numel(th),1); e(k) = h;
        gfd(k) = (fun(th+e)-fun(th-e))/(2*h);
    end
    [~,gan] = fun(th);
    fprintf('gradient check: ||g_an - g_fd||/||g_fd|| = %.3e\n', norm(gan-gfd)/norm(gfd));
end


function [theta,f,it] = bfgs_armijo(fun,theta,opt,cache)
    % fallback when the Optimization Toolbox is not installed
    n = numel(theta);
    [f,g] = fun(theta);
    Hinv = eye(n);
    if opt.disp, fprintf('iter %3d   sigma = %.10e   |g| = %.3e\n',0,f,norm(g)); end
    it = 0;
    for it = 1:opt.maxit
        d = -Hinv*g;
        if g.'*d >= 0, Hinv = eye(n); d = -g; end      % reset if not descent
        % Armijo backtracking; cap the step so Q stays well conditioned
        step = min(1, opt.max_step/norm(d)); accepted = false;
        for ls = 1:30
            th_new = theta + step*d;
            [f_new,g_new] = fun(th_new);
            if f_new <= f + 1e-4*step*(g.'*d), accepted = true; break, end
            step = step/2;
        end
        if ~accepted, it = it-1; break, end

        s = th_new-theta; y = g_new-g;
        if s.'*y > 1e-12*norm(s)*norm(y)               % BFGS update
            rho = 1/(s.'*y); V = eye(n)-rho*(s*y.');
            Hinv = V*Hinv*V.' + rho*(s*s.');
        end
        rel_dec = (f-f_new)/f;
        theta = th_new; f = f_new; g = g_new;
        cache('hist') = [cache('hist'), f];
        if opt.disp, fprintf('iter %3d   sigma = %.10e   |g| = %.3e\n',it,f,norm(g)); end
        if rel_dec < opt.tol_f || norm(g) < opt.tol_g, break, end
    end
end


function [f,grad] = objective(theta,ops,N,cache)
    % f = sigma_max[(Q (x) I_3N) M (Q^{-1} (x) I_N)],  grad = df/dtheta
    [Q,L] = theta_to_Q(theta);
    Qinv = Q\eye(3);

    Afun = @(x,tflag) scaled_apply(x,tflag,ops,Q,Qinv,N);
    sopts.tol = ops.svds_tol;
    sopts.maxit = 300;
    v0 = cache('v0');
    if ~isempty(v0), sopts.v0 = v0; end
    [u,s,v] = svds(Afun,ops.size,1,'largest',sopts);
    f = s(1,1);
    cache('v0') = v;

    if nargout < 2, return, end
    % generalized eigvec x = Q_R^{-1} v,  y = M x = Q_L^{-1} (sigma u)
    x = kron_apply(Qinv,v,N);
    y = kron_apply(Qinv,f*u,3*N);
    X = reshape(x,N,3);   Y = reshape(y,3*N,3);
    Z = Y'*Y - f^2*(X'*X);          % Z_ij = y_i^* y_j - lambda x_i^* x_j
    G = Z.';                        % d(lambda) = tr(dP G)
    GL = 2*G*L;                     % d(lambda) = Re sum conj(GL_ij) dL_ij  (x2 included)

    dl = zeros(8,1);
    dl(1) = real(GL(2,2))*L(2,2);   % a2 (log L22)
    dl(2) = real(GL(3,3))*L(3,3);   % a3 (log L33)
    dl(3) = real(GL(2,1)); dl(4) = imag(GL(2,1));
    dl(5) = real(GL(3,1)); dl(6) = imag(GL(3,1));
    dl(7) = real(GL(3,2)); dl(8) = imag(GL(3,2));
    grad = dl/(2*f);                % d(sigma) = d(lambda)/(2 sigma)
end


function [Q,L] = theta_to_Q(theta)
    L = [1, 0, 0;
         theta(3)+1i*theta(4), exp(theta(1)), 0;
         theta(5)+1i*theta(6), theta(7)+1i*theta(8), exp(theta(2))];
    Q = L';                          % Q^*Q = L L^* = P
end


function z = scaled_apply(x,tflag,ops,Q,Qinv,N)
    % notransp: (Q (x) I_3N) M (Q^{-1} (x) I_N) x
    % transp  : (Q^{-*} (x) I_N) M^* (Q^* (x) I_3N) x
    if strcmp(tflag,'notransp')
        z = kron_apply(Q, ops.apply(kron_apply(Qinv,x,N),'notransp'), 3*N);
    else
        z = kron_apply(Qinv', ops.apply(kron_apply(Q',x,3*N),'transp'), N);
    end
end


function z = kron_apply(Q,x,n)
    % (Q (x) I_n) x  without forming the Kronecker product
    z = reshape(reshape(x,n,3)*Q.',[],1);
end


%% =====================================================================
%  H_grad: full matrix
%  =====================================================================
function z = apply_full(M,x,tflag)
    if strcmp(tflag,'notransp'), z = M*x; else, z = M'*x; end
end


function Hgrad = build_H_grad_full(params)
    % H from resolvent_2D_xy is weighted: H_w = W^{1/2} H_phys W^{-1/2}.
    % Weighted gradient output: W9^{1/2} G H_phys W^{-1/2} = W9^{1/2} G W3^{-1/2} H_w.
    N = params.N;
    H = resolvent_2D_xy(params);
    Gs = [kron(params.Dx_mat,speye(params.Ny));
          kron(speye(params.Nx),params.Dy_mat);
          1i*params.kz*speye(N)];
    G = blkdiag(Gs,Gs,Gs);
    w  = params.w_2D;
    W9h = spdiags(repmat(sqrt(w),9,1),0,9*N,9*N);
    W3hi = spdiags(repmat(1./sqrt(w),3,1),0,3*N,3*N);
    Hgrad = full(W9h*(G*(W3hi*H)));
end


%% =====================================================================
%  H_grad: matrix-free
%  =====================================================================
function z = apply_H_grad_mf(x,tflag,params)
    % batched over the 3 velocity components with pagemtimes
    N = params.N; Ny = params.Ny; Nx = params.Nx; kz = params.kz;
    wh  = sqrt(params.w_2D);            % N x 1
    w3h = sqrt(params.w_all);           % 3N x 1
    Dx = params.Dx_mat; Dy = params.Dy_mat;
    if strcmp(tflag,'notransp')
        q  = H_fun(x,'notransp',params);                 % W^{1/2} u
        Uc = reshape(q./w3h,Ny,Nx,3);                    % pages = u,v,w
        gx = pagemtimes(Uc,Dx.');
        gy = pagemtimes(Dy,Uc);
        gz = 1i*kz*Uc;
        % N x 3(grad dir) x 3(component) -> [ux;uy;uz | vx;vy;vz | wx;wy;wz]
        T = cat(2,reshape(gx,N,1,3),reshape(gy,N,1,3),reshape(gz,N,1,3)).*wh;
        z = T(:);
    else
        T  = reshape(x,N,3,3).*wh;
        gx = reshape(T(:,1,:),Ny,Nx,3);
        gy = reshape(T(:,2,:),Ny,Nx,3);
        gz = reshape(T(:,3,:),Ny,Nx,3);
        % exact discrete conjugate transpose of [Dx; Dy; i kz]
        Uc = pagemtimes(gx,conj(Dx)) + pagemtimes(Dy',gy) + conj(1i*kz)*gz;
        z  = H_fun(reshape(Uc,3*N,1)./w3h,'transp',params);
    end
end


%% =====================================================================
%  Resolvent machinery (copied from main_2D_xy_resolvent_precondition_1D_resolvent.m)
%  =====================================================================
function params=prepare_params(params)
    Ny=params.Ny; Nx=params.Nx; N=params.N;
    [params.cheb_y,DM]=chebdif(Ny,2);
    params.Dy_mat=(2/params.Ly)*DM(:,:,1);
    params.Dyy_mat=(2/params.Ly)^2*DM(:,:,2);
    [~,Dx_mat]=fourdif(Nx,1);
    [~,Dxx_mat]=fourdif(Nx,2);
    params.Dx_mat=(2*pi/params.Lx)*Dx_mat;
    params.Dxx_mat=(2*pi/params.Lx)^2*Dxx_mat;

    % laminar Poiseuille base flow (change here for other base flows)
    params.U=reshape((1-params.cheb_y.^2)*ones(1,Nx),N,1);
    params.dUdy=reshape(-2*params.cheb_y*ones(1,Nx),N,1);
    params.dUdx=zeros(N,1);
    params.V=zeros(N,1);
    params.dVdx=zeros(N,1);
    params.dVdy=zeros(N,1);
    params.K_inv=zeros(N,1);

    params.U_mat=reshape(params.U,Ny,Nx);
    params.dUdx_mat=reshape(params.dUdx,Ny,Nx);
    params.dUdy_mat=reshape(params.dUdy,Ny,Nx);
    params.V_mat=reshape(params.V,Ny,Nx);
    params.dVdx_mat=reshape(params.dVdx,Ny,Nx);
    params.dVdy_mat=reshape(params.dVdy,Ny,Nx);
    params.K_inv_mat=reshape(params.K_inv,Ny,Nx);

    params.U_xave_diag=spdiags(mean(params.U_mat,2),0,Ny,Ny);
    params.V_xave_diag=spdiags(mean(params.V_mat,2),0,Ny,Ny);
    params.dUdx_xave_diag=spdiags(mean(params.dUdx_mat,2),0,Ny,Ny);
    params.dUdy_xave_diag=spdiags(mean(params.dUdy_mat,2),0,Ny,Ny);
    params.dVdx_xave_diag=spdiags(mean(params.dVdx_mat,2),0,Ny,Ny);
    params.dVdy_xave_diag=spdiags(mean(params.dVdy_mat,2),0,Ny,Ny);
    params.K_inv_xave_diag=spdiags(mean(params.K_inv_mat,2),0,Ny,Ny);

    [~,w]=clencurt(Ny-1);
    params.w=w(:);
    params.w_2D=reshape(params.w*ones(1,Nx),N,1);
    params.w_all=[params.w_2D;params.w_2D;params.w_2D];
end


function params = add_preconditioner(params)
    % Fourier-in-x block preconditioner; all Nx blocks inverted at once
    % with pageinv.  The adjoint solve uses the conjugate transpose:
    % (F^{-1} blkdiag(P_k^{-1}) F)^H = F^{-1} blkdiag(P_k^{-H}) F.
    Ny=params.Ny; Nx=params.Nx;
    wave=fourier_wavenumbers(Nx);
    if strcmp(params.preconditioner_type,'fft_1d_laplacian')
        P=zeros(Ny,Ny,Nx);
        for k=1:Nx
            lap_k=params.Dyy_mat+(-(2*pi/params.Lx)^2*wave(k)^2-params.kz^2)*eye(Ny);
            lap_k(1,:)=[1,zeros(1,Ny-1)];
            lap_k(Ny,:)=[zeros(1,Ny-1),1];
            P(:,:,k)=lap_k;
        end
    else
        P=complex(zeros(4*Ny,4*Ny,Nx));
        for k=1:Nx
            P(:,:,k)=full(build_1d_block(params,wave(k)));
        end
    end
    params.precond_factors   = pageinv(P);
    params.precond_factors_H = pagectranspose(params.precond_factors);
end


function P=build_1d_block(params,wave_k)
    Ny=params.Ny; kz=params.kz; Re=params.Re;
    I=speye(Ny); Z=sparse(Ny,Ny);
    kx=(2*pi/params.Lx)*wave_k;
    if mod(params.Nx,2)==0 && wave_k==-params.Nx/2
        Dx_k=Z;
    else
        Dx_k=1i*kx*I;
    end
    lap_k=params.Dyy_mat-(kx^2+kz^2)*I;
    U=params.U_xave_diag; V=params.V_xave_diag;
    dUdx=params.dUdx_xave_diag; dUdy=params.dUdy_xave_diag;
    dVdx=params.dVdx_xave_diag; dVdy=params.dVdy_xave_diag;
    K_inv=params.K_inv_xave_diag;
    sA=1i*params.omega*I+U*Dx_k+V*params.Dy_mat-lap_k/Re+K_inv;
    P=[sA+dUdx, dUdy, Z, Dx_k; dVdx, sA+dVdy, Z, params.Dy_mat; ...
       Z, Z, sA, 1i*kz*I; -Dx_k, -params.Dy_mat, -1i*kz*I, Z];
    for blk=0:2
        rows=blk*Ny+[1 Ny];
        P(rows,:)=0;
        P(sub2ind(size(P),rows,rows))=1;
    end
end


function wave=fourier_wavenumbers(Nx)
    N1=floor((Nx-1)/2);
    N2=(-Nx/2)*ones(rem(Nx+1,2));
    wave=[0:N1 N2 -N1:-1].';
end


function H_f=H_fun(f,tflag,params)
    % notransp: H_w f   = C L^{-1} B f
    % transp  : H_w^* f = B^* L^{-*} C^* f   (exact discrete adjoint)
    Ny=params.Ny; N=params.N; w_all=params.w_all;
    wall=[1:Ny:N, Ny:Ny:N];
    wall_vel=[wall, N+wall, 2*N+wall];
    if strcmp(tflag,'notransp')
        rhs=[f.*w_all.^(-1/2); zeros(N,1)];
        rhs(wall_vel)=0;                         % B has zero wall rows
    else
        rhs=[f.*w_all.^(1/2); zeros(N,1)];       % C^* f
    end
    [z,flag,relres] = gmres(@(u_p) L_fun(u_p,tflag,params),rhs, ...
        params.gmres_restart,params.gmres_tol,params.gmres_maxit, ...
        @(r) preconditioner_fun(r,tflag,params));
    if flag ~= 0
        warning('H_fun:gmres','GMRES flag %d, relres %.2e (%s)',flag,relres,tflag);
    end
    if strcmp(tflag,'notransp')
        H_f = w_all.^(1/2).*z(1:3*N);
    else
        zv = z(1:3*N); zv(wall_vel)=0;           % B^* z
        H_f = w_all.^(-1/2).*zv;
    end
end


function z=preconditioner_fun(rhs,tflag,params)
    Ny=params.Ny; Nx=params.Nx; N=params.N;
    if strcmp(tflag,'notransp')
        F=params.precond_factors;
    else
        F=params.precond_factors_H;
    end
    if strcmp(params.preconditioner_type,'fft_1d_laplacian')
        % velocity: same Ny x Ny block for u,v,w at each kx; pressure: identity
        R=permute(fft(reshape(rhs(1:3*N),Ny,Nx,3),[],2),[1 3 2]);   % Ny x 3 x Nx
        Z=ipermute(pagemtimes(F,R),[1 3 2]);
        z=[reshape(ifft(Z,[],2),3*N,1); rhs(3*N+1:4*N)];
    else
        % per kx: stacked [u;v;w;p] (4Ny) times the 4Ny x 4Ny block
        R=fft(reshape(rhs,Ny,Nx,4),[],2);
        R=reshape(permute(R,[1 3 2]),4*Ny,1,Nx);
        Z=permute(reshape(pagemtimes(F,R),Ny,4,Nx),[1 3 2]);
        z=reshape(ifft(Z,[],2),4*N,1);
    end
end


function L_u_p=L_fun(u_p,tflag,params)
    kz=params.kz; omega=params.omega; Ny=params.Ny; Nx=params.Nx;
    Re=params.Re; N=params.N;
    u=reshape(u_p(1:N),Ny,Nx);
    v=reshape(u_p(N+1:2*N),Ny,Nx);
    w=reshape(u_p(1+2*N:3*N),Ny,Nx);
    p=reshape(u_p(1+3*N:4*N),Ny,Nx);
    U=params.U_mat; V=params.V_mat; K_inv=params.K_inv_mat;
    dUdx=params.dUdx_mat; dUdy=params.dUdy_mat;
    dVdx=params.dVdx_mat; dVdy=params.dVdy_mat;
    if strcmp(tflag,'notransp')
        % L = P_int*L0 + P_wall  (wall rows of u,v,w replaced by identity)
        Dx=params.Dx_mat.'; Dy=params.Dy_mat;
        ux=u*Dx; vx=v*Dx; wx=w*Dx; px=p*Dx;
        uy=Dy*u; vy=Dy*v; wy=Dy*w; py=Dy*p;
        lap=@(q) params.Dyy_mat*q+q*params.Dxx_mat.'-(kz^2).*q;
        out_u=1i*omega*u+U.*ux+V.*uy-lap(u)/Re+K_inv.*u+dUdx.*u+dUdy.*v+px;
        out_v=1i*omega*v+U.*vx+V.*vy-lap(v)/Re+K_inv.*v+dVdx.*u+dVdy.*v+py;
        out_w=1i*omega*w+U.*wx+V.*wy-lap(w)/Re+K_inv.*w+1i*kz*p;
        out_p=-ux-vy-1i*kz*w;
        out_u([1 Ny],:)=u([1 Ny],:);
        out_v([1 Ny],:)=v([1 Ny],:);
        out_w([1 Ny],:)=w([1 Ny],:);
    else
        % exact discrete adjoint:  L^H = L0^H*P_int + P_wall
        ui=u; vi=v; wi=w;
        ui([1 Ny],:)=0; vi([1 Ny],:)=0; wi([1 Ny],:)=0;
        DxH=@(q) q*conj(params.Dx_mat);          % (Dx (x) Iy)^H
        DyH=@(q) params.Dy_mat'*q;               % (Ix (x) Dy)^H
        lapH=@(q) params.Dyy_mat'*q+q*conj(params.Dxx_mat)-(kz^2).*q;
        SH=@(q) -1i*omega*q+DxH(U.*q)+DyH(V.*q)-lapH(q)/Re+K_inv.*q;
        out_u=SH(ui)+dUdx.*ui+dVdx.*vi-DxH(p);
        out_v=SH(vi)+dUdy.*ui+dVdy.*vi-DyH(p);
        out_w=SH(wi)+1i*kz*p;
        out_p=DxH(ui)+DyH(vi)-1i*kz*wi;
        out_u([1 Ny],:)=out_u([1 Ny],:)+u([1 Ny],:);
        out_v([1 Ny],:)=out_v([1 Ny],:)+v([1 Ny],:);
        out_w([1 Ny],:)=out_w([1 Ny],:)+w([1 Ny],:);
    end
    L_u_p=[out_u(:);out_v(:);out_w(:);out_p(:)];
end


function H=resolvent_2D_xy(params)
    kz=params.kz; omega=params.omega; Ny=params.Ny; Nx=params.Nx;
    N=Nx*Ny; Re=params.Re;
    Ix=speye(Nx); Iy=speye(Ny); I=speye(N);
    D1=params.Dy_mat; D2=params.Dyy_mat;
    Dx2=kron(params.Dx_mat,Iy); Dy2=kron(Ix,D1);
    lplc=-kz^2*I+kron(Ix,D2)+kron(params.Dxx_mat,Iy);
    U=spdiags(params.U,0,N,N);       V=spdiags(params.V,0,N,N);
    dUdx=spdiags(params.dUdx,0,N,N); dUdy=spdiags(params.dUdy,0,N,N);
    dVdx=spdiags(params.dVdx,0,N,N); dVdy=spdiags(params.dVdy,0,N,N);
    K=spdiags(params.K_inv,0,N,N);
    S=1i*omega*I+U*Dx2+V*Dy2-(1/Re)*lplc+K;
    Z=sparse(N,N);
    L=[S+dUdx, dUdy, Z, Dx2;
       dVdx, S+dVdy, Z, Dy2;
       Z, Z, S, 1i*kz*I;
       -Dx2, -Dy2, -1i*kz*I, Z];
    wh=sqrt(params.w_2D);
    Wh=spdiags(wh,0,N,N); Whi=spdiags(1./wh,0,N,N);
    B=[blkdiag(Whi,Whi,Whi); sparse(N,3*N)];
    C=[blkdiag(Wh,Wh,Wh), sparse(3*N,N)];
    bc=[1:Ny:N, Ny:Ny:N];
    for c=0:2
        rows=c*N+bc;
        L(rows,:)=0;
        L(sub2ind(size(L),rows,rows))=1;
        B(rows,:)=0;
    end
    H=C*(L\B);
end
