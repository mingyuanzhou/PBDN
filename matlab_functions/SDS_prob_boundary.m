function [class_prob_boundary, class_ST] = SDS_prob_boundary(Beta,r_sk,X,K,S)
N = size(X,2);
class_prob_boundary =zeros(S,N);
T = size(Beta{1},2)-1;
class_ST = zeros(S,T+1);
if length(K)==1
    if K==1
        
        gt = zeros(1,T+1);
        gt(1) = 1;
        
        GT  = 0; %log(exp(1)-1);
        for t=2:T+1
            gt(t) = log(1+exp(GT)*gt(t-1));
            % gt(t) = log(1+gt(t-1)*exp(1));
            % gt(t) = GT + log(gt(t-1));
        end
        for s=1:S
            for k=1:K
                ks = k+(s-1)*K;
                T = size(Beta{ks},2)-1;
                temp = ones(1,N);
                temp = zeros(1,N);
                for t=T+1:-1:2
                    p=0.5;
                    if t==T+1
                        %ht = log(2.^(1./r_sk(ks))-1);
                        ht = (1-p).^(-1./r_sk(ks))-1;
                        % ht = (1/0.9).^(1./r_sk(ks))-1;
                    else
                        % ht = log(exp(exp(ht/GT))-1);
                        %ht = exp(ht-GT)-1;
                        
                        ht = exp(ht/exp(GT))-1;
                        
                        %                    if 1 % ht-GT<0
                        %                         ht = log(exp(exp(ht-GT))-1);
                        %                    else
                        %                         ht = exp(ht-GT) + log(1-exp(-exp(ht-GT)));
                        %                    end
                        %ht = log(exp(exp(ht)/(5))-1);
                        %ht = log(2-1);
                    end
                    %ht = 0;
                    
                    temp1 = double(bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , log(ht) - log(gt(t-1)) ));
                    
                    % temp2 = double(bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , 0));
                    
                    class_ST(s,t) = sum(temp1);
                    temp =temp + temp1;
                    
                    %temp =temp + bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , ht - log(gt(t-1)));
                    
                    %temp =temp & bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , ht - log(gt(t-1)));
                end
                class_prob_boundary(s,:) = class_prob_boundary(s,:) + temp;
            end
        end
    else
        ThetaAll=zeros(S,N);
        ThetaTest = zeros(S,N);
        for s=1:S
            for k=1:K
                ks = k+(s-1)*K;
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        ThetaTest(s,:)=logOnePlusExp(Beta{ks}(:,t+1)'*([ones(1,N);X]));
                    else
                        ThetaTest(s,:) = logOnePlusExp(Beta{ks}(:,t+1)'*([ones(1,N);X])+ log(max(ThetaTest(s,:),realmin)));
                    end
                end
                ThetaTest(s,:) = (bsxfun(@times, -r_sk(ks),ThetaTest(s,:)));
                ThetaAll(s,:) = ThetaAll(s,:)+ThetaTest(s,:);
                p=0.5;
                class_prob_boundary(s,:) = class_prob_boundary(s,:) + double( ThetaTest(s,:)<log(1-p));
            end
        end
    end
else
    ThetaAll=zeros(S,N);
    ThetaTest = zeros(S,N);
    for s=1:S
        if K(s)==1
            
            gt = zeros(1,T+1);
            gt(1) = 1;
            
            GT  = 0; %log(exp(1)-1);
            for t=2:T+1
                gt(t) = log(1+exp(GT)*gt(t-1));
                % gt(t) = log(1+gt(t-1)*exp(1));
                % gt(t) = GT + log(gt(t-1));
            end
            
            for k=1:K(s)
                ks = k+sum(K(1:s-1));
                T = size(Beta{ks},2)-1;
                temp = zeros(1,N);
                for t=T+1:-1:2
                    p=0.5;
                    if t==T+1
                        %ht = log(2.^(1./r_sk(ks))-1);
                        ht = (1-p).^(-1./r_sk(ks))-1;
                        % ht = (1/0.9).^(1./r_sk(ks))-1;
                    else
                        % ht = log(exp(exp(ht/GT))-1);
                        %ht = exp(ht-GT)-1;
                        
                        ht = exp(ht/exp(GT))-1;
                        
                        %                    if 1 % ht-GT<0
                        %                         ht = log(exp(exp(ht-GT))-1);
                        %                    else
                        %                         ht = exp(ht-GT) + log(1-exp(-exp(ht-GT)));
                        %                    end
                        %ht = log(exp(exp(ht)/(5))-1);
                        %ht = log(2-1);
                    end
                    %ht = 0;
                    
                    temp1 = double(bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , log(ht) - log(gt(t-1)) ));
                    
                    % temp2 = double(bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , 0));
                    
                    class_ST(s,t) = sum(temp1);
                    temp =temp + temp1;
                    
                    %temp =temp + bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , ht - log(gt(t-1)));
                    
                    %temp =temp & bsxfun(@ge,Beta{ks}(:,t)'*[ones(1,size(X,2));X] , ht - log(gt(t-1)));
                end
                class_prob_boundary(s,:) = class_prob_boundary(s,:) + temp;
            end
            
        else
            
            for k=1:K(s)
                ks = k+sum(K(1:s-1));
                T = size(Beta{ks},2)-1;
                for t=1:T
                    if t==1
                        ThetaTest(s,:)=logOnePlusExp(Beta{ks}(:,t+1)'*([ones(1,N);X]));
                    else
                        ThetaTest(s,:) = logOnePlusExp(Beta{ks}(:,t+1)'*([ones(1,N);X])+ log(max(ThetaTest(s,:),realmin)));
                    end
                end
                ThetaTest(s,:) = (bsxfun(@times, -r_sk(ks),ThetaTest(s,:)));
                ThetaAll(s,:) = ThetaAll(s,:)+ThetaTest(s,:);
                p=0.5;
                class_prob_boundary(s,:) = class_prob_boundary(s,:) + double( ThetaTest(s,:)<log(1-p));
            end
        end
    end
end

