function Theta0 = SDS_Theta0_V1(Beta,r_sk,X,K,S)
V=size(X,1);
N = size(X,2);
Xcovariate=[ones(1,N);X];
Theta0=0;

if length(K)==1
    Theta = zeros(K*S,N);
    for ks=1:K*S
        %s = ceil(ks/K);
        %k = ks-K*(s-1);
        if 1 %r_sk(ks)>0
            T = size(Beta{ks},2)-1;
            for t=1:T
                if t==1
                    Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                else
                    Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                end
            end
            %Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
        end
    end
    Theta0=Theta;
    %Theta = squeeze(sum(reshape(Theta,K,S,N),1));
else
    Theta = zeros(sum(K),N);
    for ks=1:sum(K+V)
        %s = ceil(ks/K);
        %k = ks-K*(s-1);
        s = 1+ nnz(ks>cumsum(K+V));
        k = ks-sum(K(1:s-1)+V);
        if r_sk(ks)>0
            if k<=K(s)
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate);
                    else
                        Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate+ log(Theta(ks,:)));
                    end
                end
            else
                T = size(Beta{ks},2)-1;
                for t=1:T
                    Theta(ks,:)= X(k-K(s),:);
%                     if t==1
%                         %Theta(ks,:)= logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate([1,1+k-K(s)],:));
%                         Theta(ks,:)= X(k-K(s),:)
%                     else
%                         Theta(ks,:) = logOnePlusExp(Beta{ks}(:,t+1)'*Xcovariate([1,1+k-K(s)],:)+ log(Theta(ks,:)));
%                     end
                end
            end
            %Theta(ks,:) = (bsxfun(@times, r_sk(ks),Theta(ks,:)));
        end
    end
    Theta0=Theta;
    %Theta = zeros(S,N);
    %for s=1:S
    %    Theta(s,:) = sum(Theta0(sum(K(1:s-1))+(1:K(s)),:),1);
    %end
end