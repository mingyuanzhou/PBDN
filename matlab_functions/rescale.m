function     [features,idx]=rescale(features,idx,option)
x_train = features(idx.train,:);
x_test = features(idx.test,:);
if nargin<3
    option=1;
end
if option ==1
    xmin = min(x_train,[],1);
    range = max(max(x_train,[],1)-min(x_train,[],1),realmin);
    features(idx.train,:) = bsxfun(@rdivide,bsxfun(@minus,x_train,xmin),range)*2-1;
    features(idx.test,:) = bsxfun(@rdivide,bsxfun(@minus,x_test,xmin),range)*2-1;
elseif option ==2
    xmin = min(x_train(:));
    range = max(x_train(:))-min(x_train(:));
    features(idx.train,:) = bsxfun(@rdivide,bsxfun(@minus,x_train,xmin),range)*2-1;
    features(idx.test,:) = bsxfun(@rdivide,bsxfun(@minus,x_test,xmin),range)*2-1;
else
    xmean = mean(x_train,1);
    xstd = max(std(x_train,1),realmin);
    features(idx.train,:) = bsxfun(@rdivide,bsxfun(@minus,x_train,xmean),xstd);
    features(idx.test,:) = bsxfun(@rdivide,bsxfun(@minus,x_test,xmean),xstd);
end
