if i<=6
    load data/benchmarks.mat
    %'banana'   'breast_cancer' 'diabetis'  'flare_solar'
    %'german'    'heart'    'image'    'ringnorm'    'splice'
    %'thyroid'    'titanic'    'twonorm'    'waveform'
    switch dataname
        case 'banana'
            benchmark = banana;
        case 'breast_cancer'
            benchmark = breast_cancer;
        case 'titanic'
            benchmark = titanic;
        case 'waveform'
            benchmark = waveform;
        case 'german'
            benchmark = german;
        case 'diabetis'
            benchmark = diabetis;
        case 'image'
            benchmark = image;
    end
    x_train = benchmark.x(benchmark.train(trial,:),:);
    t_train = benchmark.t(benchmark.train(trial,:));
    x_test  = benchmark.x(benchmark.test(trial,:),:);
    t_test  = benchmark.t(benchmark.test(trial,:));
elseif i==8
    load data/ijcnn1.mat
elseif i==9
    load data/a9a.mat
elseif i==10
    %https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets
    %figure
    N=4000;
    degree1=720;
    degree2=720;
    start = 0;
    noise = 0.8;
    deg2rad = (2*pi)/360;
    start = start * deg2rad;
    
    N1 = floor(N/2);
    N2 = N-N1;
    
    n = start + sqrt(rand(N1,1)) * degree1 * deg2rad;
    d1 = [-cos(n).*n + rand(N1,1)*noise sin(n).*n+rand(N1,1)*noise zeros(N1,1)];
    
    n = start + sqrt(rand(N1,1)) * degree2 * deg2rad;
    d2 = [cos(n).*n+rand(N2,1)*noise -sin(n).*n+rand(N2,1)*noise ones(N2,1)];
    
    data = [d1;d2];
    scatter(data(:,1), data(:,2), 12, data(:,3)); axis equal;
    title('Two spirals');
    
    x_train = data(1:2:end,1:2);
    t_train = data(1:2:end,3);
    x_test = data(2:2:end,1:2);
    t_test = data(2:2:end,3);
end


features = [x_train',x_test' ]';
species = [t_train;t_test];
idx.train = 1:length(t_train);
idx.test = length(t_train) + (1:length(t_test));
if strcmp(dataname,'breast_cancer')
    [features,idx]=rescale(features,idx,3);
end

if i==8||i==9
    features=full(features);
    idx.train = randtry:10:size(features,1);
    idx.test = 1:size(features,1);
    idx.test(idx.train)=[];
end