
%% Reproduce Figure 1
addpath matlab_functions

i=10;
dataname = 'two_spirals'






Depth = 10;
K_star=1;
randtry=1;
ParaDeep = cell(Depth,1);
MinorClassAsOne = true
%IsAsymmetric = true
IsAsymmetric = 0 %-1;
IsPlot = true;
%  IsPlot = false;
IsPrune  = true
%IsPrune  = false

trial=randtry

K=20;

K0=K;



PolyaGammaTruncation = 5;

QBOUND = 6;
qbound = 10^(-QBOUND);

Burnin  = 1000;
Collection = 1000;


CollectionStep = 50;


rng(randtry,'twister');
addpath('data')
addpath('liblinear-2.1/matlab')

loaddata

PruneIdx_K = 525:50:5000;
%softplus_reg_GaP

[Nall,V]=size(features);
N = length(idx.train);

%K=ceil(log(N));
K=20;
T0=1;


K_star = 0;

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

%[Error_SDS,ML,sample] =  softplus_reg_GaP_FOR(features,species,idx,K,T0,qbound,PolyaGammaTruncation,Burnin,Collection,CollectionStep,PruneIdx_K,IsPlot,dataname);
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
    
    
    
    save(['results/',dataname,'_PBDN_10layers.mat'],'Errors','K_NUM','depth','ParaDeep');
    
end

%% Determine the depth

V=2;
sparse_threshold=0.01;
EE=zeros(10,1);
TT=zeros(10,1);
%dataname='swissroll';
for AIC_sparse = 0:1
    
    
    
    n = 2000;
    dataname='two_spirals';
    
    T0=1;
    load(['results/',dataname,'_PBDN_10layers.mat']);
    
    
    AIC_min=inf;
    for depth=[1,2,3,4,5]
        
        if depth==1 %|| depth==2
            
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
                %AIC = 2*numel(temp1) + 2*numel(temp2) + 2*KK;
                
                %maxx = max(max(abs(temp1)),max(abs(temp2)));
                
                %AIC = AIC+2*nnz(abs(temp1)>sparse_threshold*maxx) + 2*nnz(abs(temp2)>sparse_threshold*maxx) + 2*KK;
                
                AIC = 2*nnz(abs(temp1)>sparse_threshold*max(abs(temp1))) + 2*nnz(abs(temp2)>sparse_threshold*max(abs(temp2))) + 2*KK;
                
                
            end
            %                     %AIC = log(n(i))*(ParaDeep{depth}.ML{1}.Vcovariate+1)*KK;
            
            AIC_save=AIC;
            KK_1=ParaDeep{depth}.ML{1}.Vcovariate;
            
        else
            AIC=AIC_save;
            AIC=AIC-2*KK;
            
            %                     if depth>2
            %                         AIC=AIC-(KK_1+1)*KK;
            %                     end
            
            KK_1=0;
            for side=1:2
                KK_1=KK_1+nnz(ParaDeep{depth-1}.ML{side}.r_k);
            end
            
            KK=0;
            for side=1:2
                KK=KK+nnz(ParaDeep{depth}.ML{side}.r_k);
            end
            
            
            
            %AIC = AIC+1*(KK_1+2)*KK;
            
            temp1= cell2mat(ParaDeep{depth}.ML{1}.Beta(ParaDeep{depth}.ML{1}.r_k>0));
            temp1=temp1(:,2);
            temp2 = cell2mat(ParaDeep{depth}.ML{2}.Beta(ParaDeep{depth}.ML{2}.r_k>0));
            temp2=temp2(:,2);
            
            KKK = KKK + numel(temp1) + numel(temp2);
            if AIC_sparse==0
                AIC = AIC+2*(KK_1+2)*KK;
                %                     %AIC = AIC+2*numel(temp1) + 2*numel(temp2) + 2*KK;
            else
                %maxx = max(max(abs(temp1)),max(abs(temp2)));
                
                %AIC = AIC+2*nnz(abs(temp1)>sparse_threshold*maxx) + 2*nnz(abs(temp2)>sparse_threshold*maxx) + 2*KK;
                
                AIC = AIC+2*nnz(abs(temp1)>sparse_threshold*max(abs(temp1))) + 2*nnz(abs(temp2)>sparse_threshold*max(abs(temp2))) + 2*KK;
            end
            %
            %AIC = AIC+log(n(i))*(KK_1+2)*KK;
            AIC_save=AIC;
        end
        
        
        for side=1:2
            AIC = AIC -2*N*ParaDeep{depth}.ML{1}.Error_SDS(side,3);
        end
        if AIC<AIC_min
            AIC_min=AIC
            EE=ParaDeep{depth}.ML{1}.Error_SDS(3,2);
            TT=depth;
            Cost=KKK/(ParaDeep{depth}.ML{1}.Vcovariate);
        else
            break
        end
    end
    
end


%% visulize the results
clear all
filename = 'DSN_two_spirals_subtype.gif';


for depth=1:10 %[1,2,3,4,5,6,7,8,9,10]
    h=figure(depth)
    axis tight manual
    %dataname='swissroll';
    %dataname='banana';
    dataname='two_spirals';
    T0=1
    randtry=1
    temp=depth;
    load(['results/',dataname,'_PBDN_10layers.mat']);
    depth=temp;
    i=10
    loaddata
    %[features,species,idx] = loaddata_SR(dataname,randtry);
    temp1 = (max(features(:,1)) - min(features(:,1)));
    temp2 = (max(features(:,2)) - min(features(:,2)));
    [x1,x2] = meshgrid(min(features(:,1))-temp1/5:temp1/100:max(features(:,1))+temp1/5,...
        min(features(:,2))-temp2/5:temp2/100:max(features(:,2))+temp2/5);
    xs1 = [x1(:),x2(:)];
    xs1 = xs1';
    XLIM = [(min(x1(:))),(max(x1(:)))];
    YLIM = [(min(x2(:))),(max(x2(:)))];
    
    X.train = features(idx.train,:)';
    X.test = features(idx.test,:)';
    %[species_unique,~,species_label]=unique(species,'stable');
    [species_unique,~,species_label]=unique(species);
    Label.train = species_label(idx.train);
    Label.test = species_label(idx.test);
    
    xs2 = features';
    
    if depth>1
        
        feature0=xs1';
        for tt=2:depth
            Theta01 = SDS_Theta0_V1(ParaDeep{tt-1}.ML{1}.Beta,ParaDeep{tt-1}.ML{1}.r_k,xs1,ParaDeep{tt-1}.ML{1}.K,1);
            Theta02 = SDS_Theta0_V1(ParaDeep{tt-1}.ML{2}.Beta,ParaDeep{tt-1}.ML{2}.r_k,xs1,ParaDeep{tt-1}.ML{2}.K,1);
            Theta01 = Theta01(ParaDeep{tt-1}.ML{1}.r_k>0,:);
            Theta02 = Theta02(ParaDeep{tt-1}.ML{2}.r_k>0,:);
            features = [Theta01;Theta02]';
            %             if 0
            %                 %type 1
            %                 if tt==2
            %                     features = [feature0,features];
            %                 end
            %             else
            %                 %type 3
            features = [feature0,features];
            feature0 =  [Theta01;Theta02]';
            %             end
            xs1 = features';
        end
        
        xs1 =     features';
    end
    
    
    if depth>1
        
        feature0_2=xs2';
        for tt=2:depth
            Theta01 = SDS_Theta0_V1(ParaDeep{tt-1}.ML{1}.Beta,ParaDeep{tt-1}.ML{1}.r_k,xs2,ParaDeep{tt-1}.ML{1}.K,1);
            Theta02 = SDS_Theta0_V1(ParaDeep{tt-1}.ML{2}.Beta,ParaDeep{tt-1}.ML{2}.r_k,xs2,ParaDeep{tt-1}.ML{2}.K,1);
            Theta01 = Theta01(ParaDeep{tt-1}.ML{1}.r_k>0,:);
            Theta02 = Theta02(ParaDeep{tt-1}.ML{2}.r_k>0,:);
            features_2 = [Theta01;Theta02]';
            %             if 0
            %                 %type 1
            %                 if tt==2
            %                     features = [feature0,features];
            %                 end
            %             else
            %                 %type 3
            features_2 = [feature0_2,features_2];
            feature0_2 =  [Theta01;Theta02]';
            %             end
            xs2 = features_2';
        end
        
        xs2 =     features_2';
    end
    
   
    
    
    if 1 %Vcovariate==3
        %class_prob = GBN_prob(ML.Beta,ML.R,ML.Phi,xs1,T0,S);
        for side=1:2
            ML=ParaDeep{depth}.ML{side};
            class_prob(side,:) = SDS_prob(ML.Beta,ML.r_k,xs1,ML.K,1,false);
            class_prob_boundary(side,:)=SDS_prob_boundary(ML.Beta,ML.r_k,xs1,ML.K,1);
        end
        
        class_prob_subtype=cell(1,2);
        for side=1:2
            ML=ParaDeep{depth}.ML{side};
            [class_probaaaa,~,~,~,~,temp] = SDS_prob(ML.Beta,ML.r_k,xs2,ML.K,1,false);
            %class_prob_subtype{3-side}=bsxfun(@rdivide,(temp{1}(:,idx.train)*X.train')',(sum(temp{1}(:,idx.train),2))');
            
            class_prob_subtype{side}=bsxfun(@rdivide,((temp{1}(:,idx.train).*(temp{1}(:,idx.train)>-1))*X.train')',(sum((temp{1}(:,idx.train).*(temp{1}(:,idx.train)>-1)),2))');
            %class_prob_subtype{3-side}=(bsxfun(@rdivide,temp{1}(:,idx.train),class_probaaaa(idx.train)')*X.train')'./size(X.train,2);
            % class_prob_subtype{3-side}=(bsxfun(@rdivide,temp{1}(:,idx.train),sum(temp{1}(:,idx.train),1))*X.train')'./(temp{1}(:,idx.train)*ones(size(X.train))')';
            % class_prob_subtype{3-side}=(bsxfun(@rdivide,-log(1-temp{1}(:,idx.train)),sum(-log(1-temp{1}(:,idx.train)),1))*X.train')'./(bsxfun(@rdivide,-log(1-temp{1}(:,idx.train)),sum(-log(1-temp{1}(:,idx.train)),1))*ones(size(X.train))')';
            
            %(temp{1}(:,idx.train)*ones(size(X.train))')'
        end
        
    end
    S=2;
    title_str = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)'};
    colors = {'r','b','c','g'};
    %if S==2
    %species_unique = {'Class A';'Class B'};
    %end
    
    subplot(S,3,1);
    colors1 = {'r-o','b-.*','c','g'};
    for side=1:2
        ML=ParaDeep{depth}.ML{side};
        plot(1:nnz(ML.r_k),sort(ML.r_k(ML.r_k>0),'descend'),colors1{side},'LineWidth', 1.5,  'MarkerFaceColor',colors{side} );hold on;
    end
    xlabel('Hyperplane  k')
    ylabel('Weight  r_k')
    xlim([1,20]);
    %legend('Class A as ones','Class B as ones')
    legend('Red points as ones','Blue points as ones')
    
    title(title_str{1})
    
    
    
    subplot(2,3,4);
    for side=1:2
        ML=ParaDeep{depth}.ML{side};
        if side==1
            plot(ML.LogLike,'r--','LineWidth',2)
            hold on
        else
            plot(ML.LogLike,'b:','LineWidth',2)
        end
        ylim([-0.8,0]);
    end
    
    %legend('Class 2 as zeros','Class 1 as zeros')
    %legend('Class A as ones','Class B as ones')
    legend('Red points as ones','Blue points as ones')
    
    %xlim([1,1000])
    xlabel('Iteration')
    ylabel('Average log-likelihood')
    title(title_str{4})
    %print('ICNBE_toy_MCE','-dpng','-r300');
    
    for s=1:S
        
        if 1 %Vcovariate==3
            subplot(S,3,2+3*(s-1));
            contourf(x1,x2,reshape(class_prob_boundary(s,:),size(x1,1),[]));%hold on
            
            hold on
            
            scatter(class_prob_subtype{1}(1,:),class_prob_subtype{1}(2,:), 'ro')
            scatter(class_prob_subtype{2}(1,:),class_prob_subtype{2}(2,:), 'cp')
            
            ca1 = min(double(class_prob_boundary(s,:)));
            ca2 = max(double(class_prob_boundary(s,:)));
            if ca1==ca2
                ca1=ca1-0.01;
                ca2 = ca2+0.01;
            end
            caxis([ca1,ca2]);
            
            colorbar
            xlim(XLIM);
            ylim(YLIM);
            xlabel('x_1')
            ylabel('x_2')
            title(title_str{2+3*(s-1)})
            
            subplot(S,3,3+3*(s-1));
            [CC,hh]=contourf(x1,x2,reshape(class_prob(s,:),size(x1,1),[]));%,5); %[0,0.1,0.3,0.5,0.7,0.9,1]);%hold on
            %caxis([min(class_prob(s,:)), max(class_prob(s,:))])
            caxis([0, 1])
            hold on
            %load('/Users/zhoum/Box Sync/DSN_results/swissroll1concate_adjacent.mat', 'X')
            % gscatter(X.train(1,end:-1:1), X.train(2,end:-1:1), species_unique(Label.train(end:-1:1)),'rbc','.xh',8);
            gscatter(X.train(1,:), X.train(2,:), species_unique(Label.train),'rbc','oxh',2);
            
            
            
            colorbar
            xlim(XLIM);
            ylim(YLIM);
            xlabel('x_1')
            ylabel('x_2')
            title(title_str{3+3*(s-1)})
            set(hh,'LineStyle','none')
        else
            subplot(S,4,3+4*(s-1));
            %gscatter(UU(:,1)'*X.train, UU(:,2)'*X.train, double(class_prob(s,idx.train)>0.5),'brc','x.h');%hold off
            % gscatter(UU(:,1)'*X.train, UU(:,2)'*X.train, double(class_prob0(s,idx.train)>0.5),'brc','x.h');%hold off
            scatter(UU(:,1)'*X.train, UU(:,2)'*X.train,[], class_prob_boundary(s,idx.train), 'filled');
            subplot(S,4,4+4*(s-1));
            scatter(UU(:,1)'*X.train, UU(:,2)'*X.train,[], class_prob(s,idx.train), 'filled');
            %scatter(UU(:,1)'*Xall, UU(:,2)'*Xall,[], class_prob(s,:), 'filled');
            %gscatter(UU(:,1)'*X.train, UU(:,2)'*X.train, double(class_prob(s,idx.train)>0.5),'brc','x.h');%hold off
        end
        
    end
    
    
    drawnow
    % pause
    % Capture the plot as an image
    
end

for depth=1:10
    nnz(ParaDeep{depth}.ML{1}.r_k)+nnz(ParaDeep{depth}.ML{2}.r_k)
end
figure

i=10
loaddata

filename = 'DSN_two_spirals_subtype.gif';
for depth=[11,1:10]
    h=figure(depth)
    %     if depth==1
    %         suptitle('A pair of infinite support hyperplane machines (iSHMs)')
    %
    %     elseif depth<=10
    %         suptitle(['PBDN with ' num2str(depth) ' pairs of iSHMs'])
    %
    %
    %     end
    set(gcf, 'Color', 'w');
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if depth == 11
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end