function [class_prob,log_fail_class_prob,class_prob_combine,class_prob_combine1,Theta0, X_subtype] = SDS_prob(Beta,r_sk,X,K,S,IsParfor)
if  ~exist('IsParfor','var')
    IsParfor = false;
end

N = size(X,2);
Xcovariate=[ones(1,N);X];
Theta0=0;
if IsParfor
    if length(K)==1
        Theta = zeros(K*S,N);
        parfor ks=1:K*S
            %s = ceil(ks/K);
            %k = ks-K*(s-1);
            if r_sk(ks)>0
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                    else
                        Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                    end
                end
                Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
            end
        end
        Theta0=Theta;
        Theta = squeeze(sum(reshape(Theta,K,S,N),1));
    else
        Theta = zeros(sum(K),N);
        parfor ks=1:sum(K)
            %s = ceil(ks/K);
            %k = ks-K*(s-1);
            if r_sk(ks)>0
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                    else
                        Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                    end
                end
                Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
            end
        end
        Theta0=Theta;
        Theta = zeros(S,N);
        for s=1:S
            Theta(s,:) = sum(Theta0(sum(K(1:s-1))+(1:K(s)),:),1);
        end
    end
else
    if length(K)==1
        Theta = zeros(K*S,N);
        for ks=1:K*S
            %s = ceil(ks/K);
            %k = ks-K*(s-1);
            if r_sk(ks)>0
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                    else
                        Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                    end
                end
                Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
            end
        end
        Theta0=Theta;
        Theta = squeeze(sum(reshape(Theta,K,S,N),1));
    else
        Theta = zeros(sum(K),N);
        for ks=1:sum(K)
            %s = ceil(ks/K);
            %k = ks-K*(s-1);
            if r_sk(ks)>0
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                    else
                        Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                    end
                end
                Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
            end
        end
        Theta0=Theta;
        Theta = zeros(S,N);
        for s=1:S
            Theta(s,:) = sum(Theta0(sum(K(1:s-1))+(1:K(s)),:),1);
        end
    end
end

log_fail_class_prob = -Theta;
class_prob = -expm1(-Theta);
class_prob_combine = bsxfun(@rdivide,class_prob,max(sum(class_prob,1),realmin));
class_prob_combine1 = class_prob_combine;
if S==2
    class_prob_combine1(1,:) = class_prob(1,:)/2+(1-class_prob(2,:))/2;
    class_prob_combine1(2,:) = class_prob(2,:)/2+(1-class_prob(1,:))/2;
end

X_subtype=cell(1,S);
for s=1:S
    X_subtype{s} = -expm1(-Theta0(sum(K(1:s-1))+(1:K(s)),:));
end
