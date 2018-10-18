%Reproduce the images of data subtypes shown in Table 1

name='mnist3v5' %mnist3v8, mnist4v7, mnist4v9
load(['data/',name,'.mat'])
load([name,'_PBDN_para.mat'])


features=full(x_train);
%features=full(x_test);

p_ik =  -expm1(-bsxfun(@times, r_side0, logOnePlusExp(bsxfun(@plus, WWW0{2}'*full(features'),BBB0{2}'))));
subtype = bsxfun(@times,(p_ik * features)', 1./sum(p_ik,2)');
figure;
[~,dex]=sort(r_side0,'descend');
x=DispDictionary(subtype(:,dex));
imwrite(x,[name,'_2.png'])

p_ik =  -expm1(-bsxfun(@times, r_side1, logOnePlusExp(bsxfun(@plus, WWW1{2}'*full(features'),BBB1{2}'))));
subtype = bsxfun(@times,(p_ik * features)', 1./sum(p_ik,2)');
figure;
[~,dex]=sort(r_side1,'descend');
x=DispDictionary(subtype(:,dex));
imwrite(x,[name,'_1.png'])
