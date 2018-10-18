%Reproduce results of PBDN with Gibbs sampling in Tables 2, 3, and 5.

%% Adding one layer at a time
addpath matlab_functions
warning('off','all')
datanames = {'banana', %1
    'breast_cancer', %2
    'titanic', %3
    'waveform', %4
    'german', %5
    'image', %6
    'pima_diabetes', %7
    'ijcnn1', %8
    'a9a', %9
    'diabetis', %10
    'circle', %11
    'xor', %12
    'dbmoon', %13
    'USPS3v5', %14
    'mnist2vother', %15
    'mnist2vother256', %16
    'mnist2vNo2',
    'mnist3v5',
    'USPS4vother',
    'pc_mac',
    'sat',
    'swissroll', %22
    'moon_rise'
    };

ACC=zeros(9,10);

for i=[1:6,8,9]
    if i<=6
        MaxTry=10
    else
        MaxTry=5
    end
    for randtry = 1:MaxTry
        
        clearvars -EXCEPT i datanames randtry Errors ROC PR PR1 K_NUM ACC cases AICs MaxTry
        
        [i,randtry]
        
        Depth = 5;

        ParaDeep = cell(Depth,1);
        MinorClassAsOne = true
        %IsAsymmetric = true
        IsAsymmetric = 0 %-1;
        IsPlot = true;
       % IsPlot = false;
        IsPrune  = true
        %IsPrune  = false
        
        trial=randtry

        K=20;
        T0=1;
        K_star = 0;
        
        K0=K;
        
        PolyaGammaTruncation = 5;
        
        QBOUND = 6;
        qbound = 10^(-QBOUND);
        
        Burnin  = 2500;
        Collection = 2500;
        
        CollectionStep = 50;

        
        dataname = datanames{i}
        
        rng(randtry,'twister');
        addpath('data')
        addpath('liblinear-2.1/matlab')
        
        loaddata
        
        %[features,species,idx] = loaddata_SR(dataname,randtry);
        PruneIdx_K = 525:50:5000;
        
        [Nall,V]=size(features);
        N = length(idx.train);
        
        X.train = features(idx.train,1:V)';
        X.test = features(idx.test,1:V)';
        [species_unique,~,species_label]=unique(species);
        Label.train = species_label(idx.train);
        Label.test = species_label(idx.test);
        S = length(species_unique); %the number of classes
        T = T0*ones(K,1);
        
        Yall = double(sparse(species_label,1:Nall,1,S,Nall)); %Class labels as one-hot vectors
        Y =  Yall(:,idx.train); %training class labels
        
        Xall = features';
        Xcovariate=[ones(1,size(X.train,2));X.train];
        Vcovariate=size(Xcovariate,1);
        
        side=1;
        class_prob=cell(2,1);
        ML = cell(2,1);
        sample = cell(2,1);
        LogLike_iter= cell(2,1);
        AIC_iter= cell(2,1);
        BIC_iter= cell(2,1);
        for side=1:2
            [class_prob{side},ML{side},sample{side},LogLike_iter{side}, AIC_iter{side},BIC_iter{side}]= PBDN_GibbsSampling(features,species,idx,K,T0,qbound,PolyaGammaTruncation,Burnin,Collection,CollectionStep,PruneIdx_K,IsPlot,dataname,side,K_star);
        end
        
        for side=1:2
            PredictLabel = double(class_prob{side}>0.5);
            train_error_combine =1-nnz(PredictLabel(idx.train)'==Yall(side,idx.train))/length(idx.train);
            test_error_combine = 1-nnz(PredictLabel(idx.test)'==Yall(side,idx.test))/length(idx.test);
            train_LogLike = mean(Y(side,:).*log(max(class_prob{side}(idx.train)',realmin))+(1-Y(side,:)).*log(max(1-class_prob{side}(idx.train)',realmin)));
            test_LogLike = mean(Yall(side,idx.test).*log(max(class_prob{side}(idx.test)',realmin))+(1-Yall(side,idx.test)).*log(max(1-class_prob{side}(idx.test)',realmin)));
            Error_SDS(side,:)=[train_error_combine,test_error_combine,train_LogLike,test_LogLike];
        end
        PredictLabel = double(class_prob{1}>class_prob{2});
        class_prob =(class_prob{1}+(1-class_prob{2}))/2;
        train_error_combine =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
        test_error_combine = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
        train_LogLike = mean(Y(1,:).*log(max(class_prob(idx.train)',realmin))+(1-Y(1,:)).*log(max(1-class_prob(idx.train)',realmin)));
        test_LogLike = mean(Yall(1,idx.test).*log(max(class_prob(idx.test)',realmin))+(1-Yall(1,idx.test)).*log(max(1-class_prob(idx.test)',realmin)));
        
        Error_SDS(3,:) = [train_error_combine,test_error_combine,train_LogLike,test_LogLike];
        
        Error_SDS
        Errors(i,randtry,1)=Error_SDS(3,2);
        
        Errors
         
        Knumber = nnz(ML{1}.r_k)+nnz(ML{2}.r_k);
        K_NUM(i,randtry,1) = Knumber;
        ML{1}.Error_SDS=Error_SDS;
        ParaDeep{1}.ML = ML;
        K_NUM
        
        %AICs(i,randtry,1) = ML{1}.AIC_min+ML{2}.AIC_min;
        
        
        feature0=features;
        
        for depth=2:Depth
            
            Xall = features';
            
            if depth==2
                KKK=size(features,2):-1:1;
            else
                KKK = K_NUM(i,randtry,depth-2):-1:1;
            end
            Theta01 = SDS_Theta0_V1(ML{1}.Beta,ML{1}.r_k,Xall,ML{1}.K,1);
            Theta02 = SDS_Theta0_V1(ML{2}.Beta,ML{2}.r_k,Xall,ML{2}.K,1);
            Theta01 = Theta01(ML{1}.r_k>0,:);
            Theta02 = Theta02(ML{2}.r_k>0,:);
            
            features = [Theta01;Theta02]';
            features = [feature0,features];
            feature0 =  [Theta01;Theta02]';
            
            class_prob=cell(2,1);
            ML = cell(2,1);
            sample = cell(2,1);
            LogLike_iter= cell(2,1);
            AIC_iter= cell(2,1);
            BIC_iter= cell(2,1);
            for side=1:2
                [class_prob{side},ML{side},sample{side},LogLike_iter{side}, AIC_iter{side},BIC_iter{side}] =  PBDN_GibbsSampling(features,species,idx,K,T0,qbound,PolyaGammaTruncation,Burnin,Collection,CollectionStep,PruneIdx_K,IsPlot,dataname,side,K_star);
            end
            
            for side=1:2
                PredictLabel = double(class_prob{side}>0.5);
                train_error_combine =1-nnz(PredictLabel(idx.train)'==Yall(side,idx.train))/length(idx.train);
                test_error_combine = 1-nnz(PredictLabel(idx.test)'==Yall(side,idx.test))/length(idx.test);
                train_LogLike = mean(Y(side,:).*log(max(class_prob{side}(idx.train)',realmin))+(1-Y(side,:)).*log(max(1-class_prob{side}(idx.train)',realmin)));
                test_LogLike = mean(Yall(side,idx.test).*log(max(class_prob{side}(idx.test)',realmin))+(1-Yall(side,idx.test)).*log(max(1-class_prob{side}(idx.test)',realmin)));
                Error_SDS(side,:)=[train_error_combine,test_error_combine,train_LogLike,test_LogLike];
            end
            
            PredictLabel = double(class_prob{1}>class_prob{2});
            class_prob =(class_prob{1}+(1-class_prob{2}))/2;
            train_error_combine =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
            test_error_combine = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
            train_LogLike = mean(Y(1,:).*log(max(class_prob(idx.train)',realmin))+(1-Y(1,:)).*log(max(1-class_prob(idx.train)',realmin)));
            test_LogLike = mean(Yall(1,idx.test).*log(max(class_prob(idx.test)',realmin))+(1-Yall(1,idx.test)).*log(max(1-class_prob(idx.test)',realmin)));
            
            Error_SDS(3,:) = [train_error_combine,test_error_combine,train_LogLike,test_LogLike];
            
            Error_SDS
            Errors(i,randtry,depth)=Error_SDS(3,2);
            
            Errors
            
            Knumber = nnz(ML{1}.r_k)+nnz(ML{2}.r_k);
            K_NUM(i,randtry,depth) = Knumber;
            ML{1}.Error_SDS=Error_SDS;
            ParaDeep{depth}.ML = ML;
            K_NUM
%             AICs(i,randtry,depth) = ML{1}.AIC_min+ML{2}.AIC_min;
%             
%             for dd=1:depth-1
%                 AICs(i,randtry,depth)=AICs(i,randtry,depth)+...
%                     +2*ParaDeep{dd}.ML{1}.K*(ParaDeep{dd}.ML{1}.Vcovariate+0)+...
%                     2*ParaDeep{dd}.ML{2}.K*(ParaDeep{dd}.ML{2}.Vcovariate+0);
%             end
%             
%             AICs
            save(['results/',dataname,'_PBDN_5layers_', num2str(randtry),'.mat'])
        end
        
    end
    mean(Errors,2)
    
end


%% AIC or AIC_\epsilon can be used to determine the depth (and hence could help terminate the greedy layer-wise training procedure)
n = [400 200 150 400 700 1300 0 (49990+91701)/10 (32561+16281)/10 ]
V = [2,9,3,21,20,18,0, 22,123]

sparse_threshold  = 1e-2;
for AIC_sparse=0:1
    EE=zeros(9,10);
    TT=zeros(9,10);
    Cost = zeros(9,10);
    for i=[1:6,8,9]
        if i<=6
            MaxTry=10;
        else
            MaxTry=5;
        end
        for randtry=1:MaxTry
            dataname = datanames{i};
            T0=1;
            load(['results/',dataname,'_PBDN_5layers_', num2str(randtry),'.mat'])
            
            AIC_min=inf;
            for depth=[1,2,3,4,5]
                
                if depth==1 
                    
                    KK=0;
                    for side=1:2
                        KK=KK+nnz(ParaDeep{depth}.ML{side}.r_k);
                    end
                    
                    KKK = (ParaDeep{depth}.ML{1}.Vcovariate)*KK;
                    
                    if AIC_sparse==0
                        AIC = 2*(ParaDeep{depth}.ML{1}.Vcovariate+1)*KK;
                    else
                        temp1= cell2mat(ParaDeep{depth}.ML{1}.Beta(ParaDeep{depth}.ML{1}.r_k>0));
                        temp1=temp1(:,2);
                        temp2 = cell2mat(ParaDeep{depth}.ML{2}.Beta(ParaDeep{depth}.ML{2}.r_k>0));
                        temp2=temp2(:,2);
                        AIC = 2*nnz(abs(temp1)>sparse_threshold*max(abs(temp1))) + 2*nnz(abs(temp2)>sparse_threshold*max(abs(temp2))) + 2*KK;
                    end                    
                    AIC_save=AIC;
                    KK_1=ParaDeep{depth}.ML{1}.Vcovariate;
                else
                    AIC=AIC_save;
                    AIC=AIC-2*KK;
                    KK_1=0;
                    for side=1:2
                        KK_1=KK_1+nnz(ParaDeep{depth-1}.ML{side}.r_k);
                    end
                    KK=0;
                    for side=1:2
                        KK=KK+nnz(ParaDeep{depth}.ML{side}.r_k);
                    end
                    temp1= cell2mat(ParaDeep{depth}.ML{1}.Beta(ParaDeep{depth}.ML{1}.r_k>0));
                    temp1=temp1(:,2);
                    temp2 = cell2mat(ParaDeep{depth}.ML{2}.Beta(ParaDeep{depth}.ML{2}.r_k>0));
                    temp2=temp2(:,2);
                    
                    KKK = KKK + numel(temp1) + numel(temp2);
                    if AIC_sparse==0
                        AIC = AIC+2*(KK_1+2)*KK;
                    else
                        AIC = AIC+2*nnz(abs(temp1)>sparse_threshold*max(abs(temp1))) + 2*nnz(abs(temp2)>sparse_threshold*max(abs(temp2))) + 2*KK;
                    end
                    AIC_save=AIC;
                end
                for side=1:2
                    AIC = AIC -2*n(i)*ParaDeep{depth}.ML{1}.Error_SDS(side,3);
                end
                if AIC<AIC_min
                    AIC_min=AIC;
                    EE(i,randtry)=ParaDeep{depth}.ML{1}.Error_SDS(3,2);
                    TT(i,randtry)=depth;
                    Cost(i,randtry)=KKK/(ParaDeep{depth}.ML{1}.Vcovariate);
                else
                    break
                end
            end
        end
    end
    
    if AIC_sparse==0
        AIC_Errors = EE;
        AIC_K = Cost;
        AIC_TT = TT;
    else
        AIC_sparse_Errors = EE;
        AIC_sparse_K = Cost;
        AIC_sparse_TT = TT;
    end
end

for i=[1:6,8,9]
    if i<=6
        MaxTry=10;
    else
        MaxTry=5;
    end
    for randtry=1:MaxTry
        dataname = datanames{i};
        T0=1;
        load(['results/',dataname,'_PBDN_5layers_', num2str(randtry),'.mat'])
        KKK=0;
        for depth=1:5
            PBDN_Errors(i,randtry,depth)=ParaDeep{depth}.ML{1}.Error_SDS(3,2);
            
            temp1= cell2mat(ParaDeep{depth}.ML{1}.Beta(ParaDeep{depth}.ML{1}.r_k>0));
            temp1=temp1(:,2);
            temp2 = cell2mat(ParaDeep{depth}.ML{2}.Beta(ParaDeep{depth}.ML{2}.r_k>0));
            temp2=temp2(:,2);
            KKK = KKK + numel(temp1) + numel(temp2);
            PBDN_K(i,randtry,depth) = KKK/(ParaDeep{depth}.ML{1}.Vcovariate);
        end
        
    end
end


save results/PBDN_Gibbs.mat AIC_Errors AIC_K AIC_sparse_Errors AIC_sparse_K PBDN_Errors PBDN_K AIC_TT AIC_sparse_TT




