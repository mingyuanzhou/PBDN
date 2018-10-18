function [class_prob,ML,sample,LogLike_iter, AIC_iter,BIC_iter] =  PBDN_GibbsSampling(features,species,idx,K,T0,qbound,PolyaGammaTruncation,Burnin,Collection,CollectionStep,PruneIdx_K,IsPlot,dataname,side,K_star)


%% data preparation

%eps=1e-6;
[Nall,V]=size(features);
N = length(idx.train);


X.train = features(idx.train,1:V)';
X.test = features(idx.test,1:V)';
%[species_unique,~,species_label]=unique(species,'stable');
[species_unique,~,species_label]=unique(species);
Label.train = species_label(idx.train);
Label.test = species_label(idx.test);
S = length(species_unique); %the number of classes
T = T0*ones(K,1); %allow each expert to have its own number of expert criteria

Yall = double(sparse(species_label,1:Nall,1,S,Nall)); %Class labels as one-hot vectors
Y =  Yall(:,idx.train); %training class labels

Xall = features';
Xcovariate=[ones(1,size(X.train,2));X.train];
Vcovariate=size(Xcovariate,1);

Yall=Yall(side,:);
Y=Y(side,:);

[ii,jj,M] = find(Y);
iijj=Y>0;

if IsPlot
    %figure
    Plot_Decision_Boundary = true;
    if V==2 && Plot_Decision_Boundary
        %plot classificaiton probability map for two dimensional data
        temp1 = (max(features(:,1)) - min(features(:,1)));
        temp2 = (max(features(:,2)) - min(features(:,2)));
        [x1,x2] = meshgrid(min(features(:,1))-temp1/5:temp1/100:max(features(:,1))+temp1/5,...
            min(features(:,2))-temp2/5:temp2/100:max(features(:,2))+temp2/5);
        xs1 = [x1(:),x2(:)];
        xs1 = xs1';
        XLIM = [(min(x1(:))),(max(x1(:)))];
        YLIM = [(min(x2(:))),(max(x2(:)))];
    end
    
    if V==2
        UU=eye(2);
    else
        %plot the two-dimensional projections of high dimensional data
        [UU,SS,VV]=svds(X.train,2);
        UU = UU*SS;
    end
    Colors = {'r-','g-.','b:','m--','c-','k-.'};
end

%% Initilization
text = [];
fprintf('\n Iteration: ');

hyper_a = 1e-6*ones(T0+1,1);

gamma0=1;
c0 = 1;
q = cell(sum(K),1);
Beta = cell(sum(K),1);
Psi = cell(sum(K),1);
r_k = ones(sum(K),1)/K(1);
Theta=cell(sum(K),1);
q_dot_k = zeros(sum(K),1);
DiagIdx = sparse(1:Vcovariate,1:Vcovariate,true);


M_i = cell(sum(K),1);
for k=1:K
    q{k} = ones(N,T0+1);
    Beta{k} = zeros(Vcovariate,T0+1);
    Psi{k} = zeros(N,T0+1);
    for t=2:T0+1
        q{k}(:,t) = logOnePlusExp(Psi{k}(:,t)+log(max(q{k}(:,t-1),realmin)));
    end
    q_dot_k(k) = sum(q{k}(:,T0+1));
    M_i{k} = zeros(N,T0+1);
end


maxIter = Burnin + Collection;
ML.class_prob = 0;
ML.class_prob_boundary = 0;
sample.r_k =cell(0);
sample.Beta = cell(0);
ML.LogLike = zeros(1,maxIter);
ML.K = K;
sample.K=[];

hyper_B=zeros(T0+1,K)+1e-6;
LogLike_iter = zeros(1,maxIter);
AIC_iter=zeros(1,maxIter);
BIC_iter=zeros(1,maxIter);
%% Gibbs sampling
for iter=1:maxIter
    
    %% Sample m_i and m_{ik}^{(1)}
    
    for k=1:K
        %% Downward sample theta_{k}^{(t)}
        for t=T0:-1:1
            if t==T0
                shape = r_k(k);
            else
                shape = Theta{k}(:,t+1);
            end
            Theta{k}(:,t) = randg(bsxfun(@plus,shape,M_i{k}(:,t))).*(-expm1(-q{k}(:,t+1)));
            Theta{k}(:,t) = Theta{k}(:,t)./max(q{k}(:,t),qbound);
        end
    end
    
    Theta_mat = zeros(nnz(iijj),K);
    for k=1:K
        Theta_mat(:,k) = Theta{k}(iijj,1);
    end
    Rate = sum(Theta_mat, 2);
    M = truncated_Poisson_rnd(Rate);
    M_k = mnrnd_mex(sparse(M), Theta_mat);
    
    for k=1:K
        M_i{k}(iijj,1) = M_k(:,k);
    end
    L_k = zeros(K,1);
    
    %% update expert parameters, embarrassingly parallel
    for k=1:K
        shape=0;
        tdex = true(N,T0+1);
        %% Upward sample beta_{k}^{(t)}
        for t=1:T0
            if t==T0
                shape = r_k(k)*ones(N,1);
            else
                shape = Theta{k}(:,t+1);
            end
            if t==1
                tdex(:,t) =true(N,1);
                loglogtemp= zeros(N,1);
            else
                tdex(:,t) = tdex(:,t-1) & q{k}(:,t)>0;
                loglogtemp = log(q{k}(tdex(:,t),t));
                M_i{k}(iijj,t) = CRT_matrix(M_i{k}(iijj,t-1),max(Theta{k}(iijj,t),1*realmin));
            end
            if iter>1
                alpha = randg(hyper_a(t+1)+1/2*ones(Vcovariate,1))./(hyper_B(t+1,k)+(Beta{k}(:,t+1)).^2/2);
                hyper_B(t+1,k) = randg(1e-0+ hyper_a(t+1)*Vcovariate)./(1e-0+ sum(alpha));
            else
                alpha = 10;
            end
            
            omgea_sik = PolyaGamRnd_Gam(M_i{k}(tdex(:,t),t)+shape(tdex(:,t)),Psi{k}(tdex(:,t),t+1)+loglogtemp,PolyaGammaTruncation);
            Xt= Xcovariate(:,tdex(:,t));
            Nt = nnz(tdex(:,t));
            cov_Xt=Xt*sparse(1:Nt,1:Nt,omgea_sik)*Xt';
            cov_Xt(DiagIdx) = cov_Xt(DiagIdx) + max(alpha,1e-3);
            %cov_Xt(DiagIdx) = max(cov_Xt(DiagIdx) + alpha,1e-3);
            
            [invchol,errMSG] = chol(cov_Xt);
            count_0=0;
            while errMSG~=0
                %cov_Xt =  nearestSPD(cov_Xt);
                cov_Xt(DiagIdx) =  cov_Xt(DiagIdx) + 10^(count_0-6);
                [invchol,errMSG] = chol(cov_Xt);
                count_0 = count_0+1
                flag=1
            end
            %inverse cholesky of the covariance matrix
            %invchol = inv(invchol);
            invchol = invchol\speye(Vcovariate);
            
            mu = Xt*(0.5*(M_i{k}(tdex(:,t),t)-shape(tdex(:,t))) - omgea_sik.*loglogtemp);
            %if q{k}(i,t)=0, then log(q{k}(i,t))=-infty and hence there is no need to sample the corresponding Polya-Gamma random variable
            mu = mu + Xcovariate(:,~tdex(:,t))*M_i{k}(~tdex(:,t),t);
            
            Beta{k}(:,t+1)= invchol*(randn(Vcovariate,1) + invchol'*mu);
            
            Psi{k}(:,t+1) = Beta{k}(:,t+1)'*Xcovariate;
            q{k}(:,t+1) = logOnePlusExp(Psi{k}(:,t+1)+log(q{k}(:,t)));
        end
        q_dot_k(k) = sum(q{k}(:,T0+1));
        M_i{k}(iijj,T0+1) = CRT_matrix(M_i{k}(iijj,T0),max(ones(nnz(iijj),1)*r_k(k),1*realmin));
        L_k(k) = sum(M_i{k}(:,T0+1));
        
        %r_k(k) = randg(gamma0/K+L_k(k)).*q_dot_k(k)./(c0+q_dot_k(k))./max(q_dot_k(k),qbound);
        r_k(k) = randg(gamma0/K+L_k(k))./(c0+q_dot_k(k));
        
    end
    
    
    %r_k1 = r_k;
    %% Calcluate training classification probabilities and loglikelihoods
    IsParfor=false;
    [class_prob_train,log_fail_class_prob,~,class_prob_combine] = SDS_prob(Beta,r_k.*double(L_k>0),X.train,K,1,IsParfor);
    LogLike = sum(Y.*log(max(class_prob_train',realmin))+(1-Y).*log(max(1-class_prob_train',realmin)),2)/N;
    LogLike_iter(iter)=LogLike;
    %AIC_iter(iter)=2*(Vcovariate+1)*nnz(L_k)-2*LogLike*N;
    %BIC_iter(iter)=log(N)*(Vcovariate+1)*nnz(L_k)-2*LogLike*N;
    ML.LogLike(iter) = LogLike;
    if IsPlot
        fprintf(repmat('\b',1,length(text)));
        text = sprintf('%d',iter);
        fprintf(text, iter);
    end
    
    %% Save the ML samples (based on training data)
    if iter>=100
        
        if iter==100
            ML.class_log_prob_max = -inf;
            %ML.class_log_prob_max = inf;
            ML.AIC_min = inf;
            ML.r_k=r_k;
            ML.Beta=Beta;
            ML.K=K;
            ML.Cnorm=zeros(T0+1,max(K));
            
        end
        
        if ML.LogLike(iter) >=ML.class_log_prob_max
            %if AIC_iter(iter) <=ML.AIC_min
            ML.class_log_prob_max = ML.LogLike(iter);
            ML.AIC_min =AIC_iter(iter);
            ML.K = K;
            ML.L_k = L_k;
            ML.Vcovariate=Vcovariate;
            %ML.r_k = r_k.*double(L_k>0|r_k>=eps);
            ML.r_k = r_k.*double(L_k>0);
            ML.Beta = Beta;
            ML.LogLike_Train = LogLike*N;
            
            for k=1:K
                for tt=1:T0+1
                    ML.Cnorm(tt,k)=sum(M_i{k}(:,tt) ) ;
                end
            end
            
        end
        
    end
    
    
    %if iter>500 && min(AIC_iter(iter-500:iter)) > ML.AIC_min
    if 0 %iter>500 && max(ML.LogLike(iter-500:iter)) < ML.class_log_prob_max
        [class_prob,log_fail_class_prob] = SDS_prob(ML.Beta,ML.r_k,Xall,ML.K,1,false);
        subplot(1,3,1);plot(AIC_iter(1:iter));
        subplot(1,3,2);plot(BIC_iter(1:iter));
        subplot(1,3,3);plot(LogLike_iter(1:iter));
        drawnow
        return;
    end
    
    if  mod(iter,CollectionStep)==0
        
        if 0 %iter>Burnin
            sample.r_k{end+1}=r_k; %.*double(activedex(:));
            sample.Beta{end+1}=Beta;
            
        end
        sample.K(end+1)=K;
        if IsPlot
            subplot(1,2,1)
            plot(sort(r_k,'descend'))
            subplot(1,2,2)
            plot(sample.K)
            %   plot_softplus_MCMC_adaptiveK
            %disp('\n');
            drawnow
        end
    end
    
    
    if iter>1
        c0 =randg(gamma0+1)./(1+sum(r_k));
        aa = CRT_sum_mex_v1(L_k,gamma0/K );
        p_tilde_k = q_dot_k./(q_dot_k+c0);
        gamma0 = randg(0.01 +  aa)/(0.01-1/K *sum( log1p(-p_tilde_k)));
    end
    
    dexk2 = find(L_k==0);
    [~,dexk3]=sort(r_k(dexk2),'descend');
    %if length(dexk2)>K_star
    if mod(iter,200)==0
        dex_delete = dexk2(dexk3(1:end));
        L_k(dex_delete)=[];
        Beta(dex_delete)=[];
        M_i(dex_delete)=[];
        r_k(dex_delete)=[];
        Theta(dex_delete)=[];
        Psi(dex_delete)=[];
        q(dex_delete)=[];
        q_dot_k(dex_delete)=[];
        hyper_B(:,dex_delete)=[];
        K=length(L_k);
    end
end
[class_prob,log_fail_class_prob] = SDS_prob(ML.Beta,ML.r_k,Xall,ML.K,1,false);


