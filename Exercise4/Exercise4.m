%% Load pretrained model
net = load('imagenet-vgg-f.mat');

%% Load training dataset
load inria_train
ytrain=labels;
Nims_train=size(ims,2);

%% Extract representation as 18th layer in CNN

ims_train = zeros(4096,Nims_train);
l=18;
for i=1:Nims_train
    im_ = single(ims{i}) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);
    res = vl_simplenn(net, im_);
    ims_train(:,i) = squeeze(gather(res(l+1).x));
end

%% Normalization

[Xtrain,Xtrain_mean,Xtrain_std]=zscore(ims_train);

%% Train SVM
lambda=0.1;
nepochs=60;

[ w,b ] = TrainSVM( Xtrain, ytrain, lambda, nepochs, 2 );


%% Test accuracy of SVM - Load test dataset

load inria_test
ytest=labels;
Nims_test=size(ims,2);

%% Extract representation as 18th layer in CNN

ims_test = zeros(4096,Nims_test);
l=18;
for i=1:Nims_test
    im_ = single(ims{i}) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);
    res = vl_simplenn(net, im_);
    ims_test(:,i) = squeeze(gather(res(l+1).x));
end
%% Normalization

Xtest=(ims_test-mean(Xtrain_mean))/mean(Xtrain_std);


%% Classify 
class=sign((w'*Xtest+b));
acc=sum(class==ytest')/Nims_test;
disp(['Accuracy rate: ',num2str(acc)])


