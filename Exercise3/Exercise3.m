%% Load training set
clear y ims
load pedestrian_training
ytrain=y;
Nims_train=size(ims,2);

%% Extract HOG descriptor of images in the training set

hog_ims_train = zeros(9*5*31,Nims_train);
for i=1:Nims_train
    hog = vl_hog(im2single(ims{i}),4);
    hog_ims_train(:,i)=hog(:);
end

%% Normalization

[Xtrain,Xtrain_mean,Xtrain_std]=zscore(hog_ims_train);

%% Train SVM
lambda=0.0001;
nepochs=40;

[ w,b ] = TrainSVM( Xtrain, ytrain, lambda, nepochs, 2 );

%% Test accuracy - Load test set
disp('Test')
clear ims y
load pedestrian_test
ytest=y;
Nims_test=size(ims,2);

%% Extract HOG descriptor of images in the test set

hog_ims_test = zeros(9*5*31,Nims_test);
for i=1:Nims_test
    hog = vl_hog(im2single(ims{i}),4);
    hog_ims_test(:,i)=hog(:);
end

%% Normalization

Xtest=(hog_ims_test-mean(Xtrain_mean))/mean(Xtrain_std);

%% Classify

class=sign((w'*Xtest+b));
acc=sum(class==ytest')/Nims_test;
disp(['Accuracy rate: ',num2str(acc)])
disp('==============')

%% Search of a good value of lambda with validation set
disp('Search values of lambda');

decs=6;
lambda=logspace(-1,-decs,decs);

disp('Possible values:')
disp(lambda)

%% Load validation set
clear ims y
load pedestrian_validation
yval=y;
Nims_val=size(ims,2);

%% Extract HOG descriptor of images in the validation set

hog_ims_val = zeros(9*5*31,Nims_val);
for i=1:Nims_val
    hog = vl_hog(im2single(ims{i}),4);
    hog_ims_val(:,i)=hog(:);
end

%% Normalization

Xval=(hog_ims_val-mean(Xtrain_mean))/mean(Xtrain_std);

%% Train for each lambda and test accuracy on val set

W=zeros(size(Xtrain,1),decs);
b=zeros(decs,1);
acc=zeros(decs,1);
for i=1:decs
    [W(:,i),b(i)]=TrainSVM( Xtrain, ytrain, lambda(i));
    class=sign((W(:,i)'*Xval+b(i)));
    acc(i)=sum(class==yval')/Nims_val;
    disp(['Lambda: ',num2str(lambda(i)),' - Accuracy rate: ',num2str(acc(i))])
end

max_acc_lambda=lambda(max(acc)==acc);
disp('Best lambda:')
disp(max_acc_lambda)

%% Test final lambda on test set
disp('Final Test with selected lambda')
disp('Train with training and validation set')
[Xfinal,Xfinal_mean,Xfinal_std]=zscore([hog_ims_train,hog_ims_test]);
[ w,b ] = TrainSVM( Xfinal, [ytrain;yval]);

Xtest=(hog_ims_test-mean(Xfinal_mean))/mean(Xfinal_std);

class=sign((w'*Xtest+b));
acc=sum(class==ytest')/Nims_test;
disp(['Accuracy rate: ',num2str(acc)])



